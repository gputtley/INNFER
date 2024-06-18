import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import bayesflow as bf
import tensorflow as tf
import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from data_loader import DataLoader
from innfer_trainer import InnferTrainer
from plotting import plot_histograms
from scipy import integrate
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as roc_curve_auc
from itertools import product
from functools import partial

from data_processor import DataProcessor
from useful_functions import GetYName, MakeDirectories


class Network():
  """
  Network class for building and training Bayesian neural networks.
  """
  def __init__(self, X_train, Y_train, wt_train, X_test, Y_test, wt_test, options={}):
    """
    Initialize the Network instance.

    Args:
        X_train (str): Path to the training data for features.
        Y_train (str): Path to the training data for target variables.
        wt_train (str): Path to the training data for weights.
        X_test (str): Path to the test data for features.
        Y_test (str): Path to the test data for target variables.
        wt_test (str): Path to the test data for weights.
        options (dict): Additional options for customization.
    """
    # Coupling parameters
    self.coupling_design = "affine"
    self.permutation = "learnable"
    self.num_coupling_layers = 8

    # affine parameters
    self.affine_units_per_dense_layer = 128
    self.affine_num_dense_layers = 2
    self.affine_activation = "relu"
    self.affine_dropout = True
    self.affine_mc_dropout = True
    self.affine_dropout_prob = 0.05

    # spline parameters
    self.spline_units_per_dense_layer = 128
    self.spline_num_dense_layers = 2
    self.spline_activation = "relu"
    self.spline_dropout = True
    self.spline_mc_dropout = True
    self.spline_dropout_prob = 0.05
    self.spline_bins = 16

    # Training parameters
    self.early_stopping = True
    self.epochs = 15
    self.batch_size = 2**6
    self.learning_rate = 1e-3
    self.permutation = "learnable"
    self.optimizer_name = "Adam" 
    self.lr_scheduler_name = "ExponentialDecay"
    self.lr_scheduler_options = {} 
    self.active_learning = False
    self.active_learning_options = {}
    self.resample = False

    # Other
    self.disable_tqdm = False
    self.use_wandb = False

    # Running parameters
    self.plot_loss = True
    self.plot_lr = True
    self.plot_dir = "plots"

    # Data parameters
    self.data_parameters = {}

    self._SetOptions(options)

    # Model and trainer store
    self.inference_network = None
    self.amortizer = None
    self.trainer = None
    self.lr_scheduler = None
    
    # Data parquet files
    self.X_train = DataLoader(X_train, batch_size=self.batch_size)
    self.Y_train = DataLoader(Y_train, batch_size=self.batch_size)
    self.wt_train = DataLoader(wt_train, batch_size=self.batch_size)
    self.X_test = DataLoader(X_test, batch_size=self.batch_size)
    self.Y_test = DataLoader(Y_test, batch_size=self.batch_size)
    self.wt_test = DataLoader(wt_test, batch_size=self.batch_size)

    # Other
    self.fix_1d = (self.X_train.num_columns == 1)
    self.adaptive_lr_scheduler = None

  def _SetOptions(self, options):
    """
    Set options for the Network instance.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def BuildModel(self):
    """
    Build the conditional invertible neural network model.
    """
    latent_dim = self.X_train.num_columns

    # fix 1d latent space for spline and affine
    if self.fix_1d:
      print("WARNING: Running fix for 1D latent space. Code may take longer to run.")
      latent_dim = 2

    # Print warning about mc_dropout:
    if (self.coupling_design == "interleaved" and (self.affine_mc_dropout or self.spline_mc_dropout)) or (self.coupling_design == "affine" and self.affine_mc_dropout) or (self.coupling_design == "spline" and self.spline_mc_dropout):
      print("WARNING: Using MC dropout will give variations in the Probability output.")

    affine_settings = {
      "dense_args": dict(
        kernel_regularizer=None,
        units=self.affine_units_per_dense_layer, 
        activation=self.affine_activation), 
      "dropout": self.affine_dropout,
      "mc_dropout": self.affine_mc_dropout,
      "num_dense": self.affine_num_dense_layers,
      "dropout_prob": self.affine_dropout_prob,
      }

    spline_settings = {
      "dense_args": dict(
        kernel_regularizer=None, 
        units=self.spline_units_per_dense_layer, 
        activation=self.spline_activation), 
      "dropout": self.spline_dropout,
      "mc_dropout": self.spline_mc_dropout,
      "num_dense": self.spline_num_dense_layers,
      "dropout_prob": self.spline_dropout_prob,
      "bins": self.spline_bins
      }

    if self.coupling_design == "interleaved":
      settings = {
        "affine" : affine_settings,
        "spline" : spline_settings
      }
    elif self.coupling_design == "spline":
      settings = spline_settings
    elif self.coupling_design == "affine":
      settings = affine_settings

    self.inference_net = bf.networks.InvertibleNetwork(
      num_params=latent_dim,
      num_coupling_layers=self.num_coupling_layers,
      permutation=self.permutation,
      coupling_design=self.coupling_design,
      coupling_settings=settings,
      )

    latent_dist = None
    self.amortizer = bf.amortizers.AmortizedPosterior(self.inference_net, latent_dist=latent_dist)

  def BuildTrainer(self):
    """
    Build the trainer for training the model.
    """
    def config(forward_dict):
      out_dict = {}
      out_dict["direct_conditions"] = forward_dict["sim_data"]
      out_dict["parameters"] = forward_dict["prior_draws"]
      return out_dict

    self.trainer = InnferTrainer(
      amortizer=self.amortizer, 
      configurator=config, 
      default_lr=self.learning_rate, 
      memory=False,
    )
    self.trainer.fix_1d = self.fix_1d
    self.trainer.active_learning = self.active_learning
    self.trainer.resample = self.resample

    if self.lr_scheduler_name == "ExponentialDecay":
      self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        self.trainer.default_lr, 
        decay_rate=self.lr_scheduler_options["decay_rate"] if "decay_rate" in self.lr_scheduler_options.keys() else 0.9,
        decay_steps=int(self.X_train.num_rows/self.batch_size)
      )    
    elif self.lr_scheduler_name == "ExponentialDecayWithConstant":
      class ExponentialDecayWithConstant(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate, decay_rate, decay_steps, minimum_learning_rate):
          super(ExponentialDecayWithConstant, self).__init__()
          self.initial_learning_rate = initial_learning_rate
          self.decay_rate = decay_rate
          self.decay_steps = decay_steps
          self.minimum_learning_rate = minimum_learning_rate
        def __call__(self, step):
          learning_rate = ((self.initial_learning_rate-self.minimum_learning_rate) * (self.decay_rate**(step/self.decay_steps))) + self.minimum_learning_rate
          return learning_rate
      self.lr_scheduler = ExponentialDecayWithConstant(
          initial_learning_rate=self.trainer.default_lr,
          decay_rate=self.lr_scheduler_options["decay_rate"] if "decay_rate" in self.lr_scheduler_options.keys() else 0.9,
          decay_steps=int(self.X_train.num_rows/self.batch_size),
          minimum_learning_rate=self.lr_scheduler_options["min_lr"] if "min_lr" in self.lr_scheduler_options.keys() else 0.0001
      )
    elif self.lr_scheduler_name == "PolynomialDecay":
      self.lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
          initial_learning_rate=self.trainer.default_lr,
          decay_steps=self.lr_scheduler_options["decay_steps"] if "decay_steps" in self.lr_scheduler_options.keys() else self.epochs*int(self.X_train.num_rows/self.batch_size),
          end_learning_rate=self.lr_scheduler_options["end_lr"] if "end_lr" in self.lr_scheduler_options.keys() else 0.0,
          power=self.lr_scheduler_options["power"] if "power" in self.lr_scheduler_options.keys() else 1.0,
      )
    elif self.lr_scheduler_name == "CosineDecay":
      self.lr_scheduler = tf.keras.experimental.CosineDecay(
          initial_learning_rate=self.trainer.default_lr,
          decay_steps=self.lr_scheduler_options["decay_steps"] if "decay_steps" in self.lr_scheduler_options.keys() else self.epochs*int(self.X_train.num_rows/self.batch_size),
          alpha=self.lr_scheduler_options["alpha"] if "alpha" in self.lr_scheduler_options.keys() else 0.0
      )
    elif self.lr_scheduler_name == "NestedCosineDecay":
      class NestedCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate, outer_period, inner_period):
          super(NestedCosineDecay, self).__init__()
          self.initial_learning_rate = initial_learning_rate
          self.outer_period = outer_period
          self.inner_period = inner_period
        def __call__(self, step):
          outer_amplitude = 0.5 * (
              1 + tf.math.cos((step / self.outer_period) * np.pi)
          )
          inner_amplitude = 0.5 * (
              1 + tf.math.cos((step % self.inner_period / self.inner_period) * np.pi)
          )
          learning_rate = self.initial_learning_rate * outer_amplitude * inner_amplitude
          return learning_rate
      self.lr_scheduler = NestedCosineDecay(
        initial_learning_rate=self.trainer.default_lr,
        outer_period=self.lr_scheduler_options["outer_period"]*int(np.ceil(self.X_train.num_rows/self.batch_size)) if "outer_period" in self.lr_scheduler_options.keys() else self.epochs*int(np.ceil(self.X_train.num_rows/self.batch_size)),
        inner_period=self.lr_scheduler_options["inner_period"]*int(np.ceil(self.X_train.num_rows/self.batch_size)) if "inner_period" in self.lr_scheduler_options.keys() else 10*int(np.ceil(self.X_train.num_rows/self.batch_size)),
      )
    elif self.lr_scheduler_name == "Cyclic":
      class Cyclic(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr, min_lr, step_size):
          super(Cyclic, self).__init__()
          self.max_lr = max_lr
          self.min_lr = min_lr
          self.step_size = step_size
          self.np = np
        def __call__(self, step):
          learning_rate = (((self.max_lr-self.min_lr)/2)*(tf.math.cos(2*self.np.pi*tf.cast(step, tf.float32)/self.step_size) + 1)) + self.min_lr
          return learning_rate
      self.lr_scheduler = Cyclic(
        max_lr=self.trainer.default_lr,
        min_lr=self.lr_scheduler_options["min_lr"] if "min_lr" in self.lr_scheduler_options.keys() else 0.0,
        step_size=int(np.ceil(self.X_train.num_rows/self.batch_size))/self.lr_scheduler_options["cycles_per_epoch"] if "cycles_per_epoch" in self.lr_scheduler_options.keys() else int(np.ceil(self.X_train.num_rows/self.batch_size))/2,
      )
    elif self.lr_scheduler_name == "CyclicWithExponential":
      class CyclicWithExponential(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr, lr_shift, step_size_for_cycle, decay_rate, step_size_for_decay, offset):
          super(CyclicWithExponential, self).__init__()
          self.max_lr = max_lr
          self.lr_shift = lr_shift
          self.step_size_for_cycle = step_size_for_cycle
          self.decay_rate = decay_rate
          self.step_size_for_decay = step_size_for_decay
          self.offset = offset
          self.np = np
        def __call__(self, step):
          decayed_lr = ((self.max_lr-self.offset) * (self.decay_rate**(tf.cast(step, tf.float32)/float(self.step_size_for_decay)))) + self.offset
          cycled_lr = self.lr_shift * tf.math.cos(2*self.np.pi*tf.cast(step, tf.float32)/self.step_size_for_cycle)
          learning_rate = decayed_lr + cycled_lr
          return learning_rate
      self.lr_scheduler = CyclicWithExponential(
          max_lr = self.trainer.default_lr,
          lr_shift = self.lr_scheduler_options["lr_shift"] if "lr_shift" in self.lr_scheduler_options.keys() else 0.00035,
          step_size_for_cycle = int(np.ceil(self.X_train.num_rows/self.batch_size))/self.lr_scheduler_options["cycles_per_epoch"] if "cycles_per_epoch" in self.lr_scheduler_options.keys() else int(np.ceil(self.X_train.num_rows/self.batch_size))/0.5,
          decay_rate = self.lr_scheduler_options["decay_rate"] if "decay_rate" in self.lr_scheduler_options.keys() else 0.5,
          step_size_for_decay = self.lr_scheduler_options["decay_steps"] if "decay_steps" in self.lr_scheduler_options.keys() else int(np.ceil(self.X_train.num_rows/self.batch_size)),
          offset = self.lr_scheduler_options["offset"] if "offset" in self.lr_scheduler_options.keys() else 0.0004,
      )
    elif self.lr_scheduler_name == "AdaptiveExponential":
      self.lr_scheduler = self.trainer.default_lr
      class adaptive_lr_scheduler():
        def __init__(self, num_its=10, decay_rate=0.99):
          self.num_its = num_its
          self.decay_rate = decay_rate
          self.loss_stores = []
        def update(self, lr, loss):
          self.loss_stores.append(loss)
          if len(self.loss_stores) < self.num_its + 1:
            return lr
          else:
            rolling_ave_before = np.sum(self.loss_stores[:1])
            self.loss_stores = self.loss_stores[-1:]
            rolling_ave_after = np.sum(self.loss_stores)
            if rolling_ave_before < rolling_ave_after:
              return lr*self.decay_rate
            else:
              return lr
            
      self.adaptive_lr_scheduler = adaptive_lr_scheduler(
        num_its=self.lr_scheduler_options["num_its"] if "num_its" in self.lr_scheduler_options.keys() else 10,
        decay_rate=self.lr_scheduler_options["decay_rate"] if "decay_rate" in self.lr_scheduler_options.keys() else 0.99,
      )
    elif self.lr_scheduler_name == "AdaptiveGradient":
      self.lr_scheduler = self.trainer.default_lr
      class adaptive_lr_scheduler():
        def __init__(self, num_its=60, scaling=0.001, max_lr=0.001, min_lr=0.00000001, max_shift_fraction=0.05, grad_change=1.0):
          self.num_its = num_its
          self.scaling = scaling
          self.max_lr = max_lr
          self.min_lr = min_lr
          self.max_shift_fraction = max_shift_fraction
          self.grad_change = grad_change
          self.loss_stores = []
        def update(self, lr, loss):
          self.loss_stores.append(loss)
          if len(self.loss_stores) <= 3*self.num_its:
            return lr
          else:
            self.loss_stores = self.loss_stores[1:]
            rolling_ave_before = np.sum(self.loss_stores[:self.num_its])
            rolling_ave_middle = np.sum(self.loss_stores[self.num_its:2*self.num_its])
            rolling_ave_after = np.sum(self.loss_stores[2*self.num_its:])
            grad_before = rolling_ave_middle-rolling_ave_before
            grad_after = rolling_ave_after-rolling_ave_middle
            shift = lr*self.scaling*((grad_after/grad_before)-self.grad_change)
            min_max_shift = max(min(shift,lr*self.max_shift_fraction),-self.max_shift_fraction*self.max_shift_fraction)
            lr = min_max_shift + lr
            return max(min(lr,self.max_lr),self.min_lr)
          
      self.adaptive_lr_scheduler = adaptive_lr_scheduler(
        num_its=self.lr_scheduler_options["num_its"] if "num_its" in self.lr_scheduler_options.keys() else 40,
        scaling=self.lr_scheduler_options["scaling"] if "scaling" in self.lr_scheduler_options.keys() else 0.0002,
        max_lr=self.trainer.default_lr,
        min_lr=self.lr_scheduler_options["min_lr"] if "min_lr" in self.lr_scheduler_options.keys() else 0.0000001,
        max_shift_fraction=self.lr_scheduler_options["max_shift_fraction"] if "max_shift_fraction" in self.lr_scheduler_options.keys() else 0.01,
        grad_change=self.lr_scheduler_options["grad_change"] if "grad_change" in self.lr_scheduler_options.keys() else 0.99,
      )
    elif self.lr_scheduler_name == "GaussianNoise":
      self.lr_scheduler = self.trainer.default_lr
      class noise_lr_scheduler():
        def __init__(self, initial_learning_rate, std_percentage=0.001, num_its=10):
          self.initial_learning_rate = initial_learning_rate
          self.num_its = num_its
          self.std_percentage = std_percentage
          self.loss_stores = []
          self.lr_stores = []
        def update(self, lr, loss):
          self.loss_stores.append(loss)
          self.lr_stores.append(lr)
          if len(self.lr_stores) > self.num_its:
            self.lr_stores = self.lr_stores[1:]
            if len(self.loss_stores) > self.num_its + 1:
              self.loss_stores = self.loss_stores[1:]
            loss_grads = [self.loss_stores[ind] - self.loss_stores[ind+1] for ind in range(len(self.loss_stores)-1)]
            min_ind = loss_grads.index(max(loss_grads)) # maybe make it some weighted average instead
            lr = self.lr_stores[min_ind]
          lr += np.random.normal(loc=0.0, scale=lr*self.std_percentage)
          return lr
          
      self.adaptive_lr_scheduler = noise_lr_scheduler(
        initial_learning_rate=self.trainer.default_lr,
        std_percentage=self.lr_scheduler_options["std_percentage"] if "std_percentage" in self.lr_scheduler_options.keys() else 0.01,
        num_its=self.lr_scheduler_options["num_its"] if "num_its" in self.lr_scheduler_options.keys() else 20,
      )

    else:
      print("ERROR: lr_schedule not valid.")

    if self.optimizer_name == "Adam":
      self.optimizer = tf.keras.optimizers.Adam(self.lr_scheduler)
    elif self.optimizer_name == "AdamW":
      self.optimizer = tf.keras.optimizers.AdamW(self.lr_scheduler)
    elif self.optimizer_name == "SGD":
        self.optimizer = tf.keras.optimizers.SGD(self.lr_scheduler)
    elif self.optimizer_name == "RMSprop":
        self.optimizer = tf.keras.optimizers.RMSprop(self.lr_scheduler)
    elif self.optimizer_name == "Adadelta":
        self.optimizer = tf.keras.optimizers.Adadelta(self.lr_scheduler)
    else:
      print("ERROR: optimizer not valid.")

  def Save(self, name="model.h5"):
    """
    Save the trained model weights.

    Args:
        name (str): Name of the file to save the model weights.
    """
    MakeDirectories(name)
    self.inference_net.save_weights(name)

  def Load(self, name="model.h5", seed=42):
    """
    Load the model weights.

    Args:
        name (str): Name of the file containing the model weights.
    """
    self.BuildModel()
    X_train_batch = self.X_train.LoadNextBatch().to_numpy()
    Y_train_batch = self.Y_train.LoadNextBatch().to_numpy()
    self.X_train.batch_num = 0
    self.Y_train.batch_num = 0
    _ = self.inference_net(X_train_batch, Y_train_batch)
    self.inference_net.load_weights(name)

  def Train(self):
    """
    Train the conditional invertible neural network.
    """
    self.trainer.train_innfer(
      X_train=self.X_train,
      Y_train=self.Y_train,
      wt_train=self.wt_train,
      X_test=self.X_test,
      Y_test=self.Y_test,
      wt_test=self.wt_test,       
      epochs=self.epochs, 
      batch_size=self.batch_size, 
      early_stopping=self.early_stopping,
      optimizer=self.optimizer,
      fix_1d=self.fix_1d,
      disable_tqdm=self.disable_tqdm,
      use_wandb=self.use_wandb,
      adaptive_lr_scheduler=self.adaptive_lr_scheduler,
      active_learning=self.active_learning,
      active_learning_options=self.active_learning_options,
      resample=self.resample,
    )

    if self.plot_loss:
      plot_histograms(
        range(len(self.trainer.loss_history._total_train_loss)),
        [self.trainer.loss_history._total_train_loss, self.trainer.loss_history._total_val_loss],
        ["Train", "Test"],
        title_right = "",
        name = f"{self.plot_dir}/loss",
        x_label = "Epochs",
        y_label = "Loss"
      )

    if self.plot_lr:
      plot_histograms(
        range(len(self.trainer.lr_history)),
        [self.trainer.lr_history],
        [self.lr_scheduler_name],
        title_right = "",
        name = f"{self.plot_dir}/learning_rate",
        x_label = "Epochs",
        y_label = "Learning Rate"
      )

  def GetLoss(self, dataset="train"):
    self.BuildTrainer()
    if dataset == "train":
      loss = float(self.trainer._get_epoch_loss(self.X_train, self.Y_train, self.wt_train, 0))
    elif dataset == "test":
      loss = float(self.trainer._get_epoch_loss(self.X_test, self.Y_test, self.wt_test, 0))
    return float(loss)

  def Sample(self, Y, n_events):

    # Remove unneccessary Y components
    Y = Y.loc[:,self.data_parameters["Y_columns"]]

    # Set up Y correctly
    if len(Y) == 1:
      Y = pd.DataFrame(np.tile(Y.to_numpy().flatten(), (n_events, 1)), columns=Y.columns)
    elif len(Y) == 0:
      Y = pd.DataFrame(np.tile(np.array([]), (n_events, 1)), columns=Y.columns)


    # Set up and transform Y input
    Y_dp = DataProcessor(
      [[Y]],
      "dataset",
      options = {
        "parameters" : self.data_parameters,
      }
    )
    if len(Y.columns) != 0:
      Y  = Y_dp.GetFull(
        method="dataset",
        functions_to_apply = ["transform"]
      )

    # Set up bayesflow dictionary
    batch_data = {
      "direct_conditions" : Y.to_numpy().astype(np.float32)
    }

    # Get samples
    synth = self.amortizer.sample(batch_data, 1)[:,0,:]

    # Fix 1d couplings
    if self.fix_1d:
      synth = synth[:,0].reshape(-1,1)

    return pd.DataFrame(synth, columns=self.data_parameters["X_columns"])

  def GetHistogramMetric(self, metric=["chi_squared"], n_samples=100000, n_bins=40):

    # Set up sim data processors
    sim_train_dp = DataProcessor(
      [[self.X_train.parquet_file_name,self.Y_train.parquet_file_name,self.wt_train.parquet_file_name]],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : self.data_parameters
      }
    )
    sim_test_dp = DataProcessor(
      [[self.X_test.parquet_file_name,self.Y_test.parquet_file_name,self.wt_test.parquet_file_name]],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : self.data_parameters
      }
    )

    # Get unique Y of dataset
    if len(self.data_parameters["Y_columns"]) > 0:
      unique_values = sim_train_dp.GetFull(method="unique", functions_to_apply=["untransform"])
      unique_y_values = {k : sorted(v) for k, v in unique_values.items() if k in self.data_parameters["Y_columns"]}
      unique_y_combinations = list(product(*unique_y_values.values()))
    else:
      unique_y_combinations = [None]

    metrics = {}
    # Loop through unique Y
    for uc in unique_y_combinations:

      if uc == None:
        uc_name = "all"
        Y = pd.DataFrame([])
        selection = None
      else:
        uc_name = GetYName(uc, purpose="file")
        Y = pd.DataFrame(np.array(uc), columns=list(unique_y_values.keys()))
        selection = " & ".join([f"({k}=={uc[ind]})" for ind, k in enumerate(unique_y_values.keys())])

      # Make synthetic data processors
      synth_dp = DataProcessor(
        [[partial(self.Sample, Y)]],
        "generator",
        n_events = n_samples,
        options = {
          "parameters" : self.data_parameters
        }
      )

      # Loop through columns
      for col in self.data_parameters["X_columns"]:

        # Make histograms
        synth_hist, synth_hist_uncert, bins = synth_dp.GetFull(
          method = "histogram_and_uncert",
          functions_to_apply = ["untransform"],
          bins = n_bins,
          column = col,
          density = True,
          )
        
        sim_train_hist, sim_train_hist_uncert, bins = sim_train_dp.GetFull(
          method = "histogram_and_uncert",
          functions_to_apply = ["untransform"],
          bins = bins,
          column = col,
          density = True,
          extra_sel = selection
          )

        sim_test_hist, sim_test_hist_uncert, bins = sim_test_dp.GetFull(
          method = "histogram_and_uncert",
          functions_to_apply = ["untransform"],
          bins = bins,
          column = col,
          density = True,
          extra_sel = selection
          )

        if "chi_squared" in metric:
          # Calculate chi squared
          non_zero_uncert_bins_train = np.where((synth_hist_uncert!=0) | (sim_train_hist_uncert!=0))
          non_zero_uncert_bins_test = np.where((synth_hist_uncert!=0) | (sim_test_hist_uncert!=0))

          chi_squared_per_dof_train = np.sum((synth_hist[non_zero_uncert_bins_train]-sim_train_hist[non_zero_uncert_bins_train])**2/(synth_hist_uncert[non_zero_uncert_bins_train]**2 + sim_train_hist_uncert[non_zero_uncert_bins_train]**2)) / (len(synth_hist[non_zero_uncert_bins_train]))
          chi_squared_per_dof_test = np.sum((synth_hist[non_zero_uncert_bins_test]-sim_test_hist[non_zero_uncert_bins_test])**2/(synth_hist_uncert[non_zero_uncert_bins_test]**2 + sim_test_hist_uncert[non_zero_uncert_bins_test]**2)) / (len(synth_hist[non_zero_uncert_bins_test]))

          for tt in ["train", "test"]:
            if f"chi_squared_{tt}" not in metrics.keys():
              metrics[f"chi_squared_{tt}"] = {}
            if uc_name not in metrics[f"chi_squared_{tt}"].keys():
              metrics[f"chi_squared_{tt}"][uc_name] = {}

          metrics["chi_squared_train"][uc_name][col] = float(chi_squared_per_dof_train)
          metrics["chi_squared_test"][uc_name][col] = float(chi_squared_per_dof_test)

    return metrics

  def GetAUC(self, dataset="train"):

    print("WARNING: GetAUC involves loading a lot of data into memory. This option should preferably be run on a GPU.")

    # Set up parquet files
    if dataset == "train":
      parquet_files = [[self.X_train.parquet_file_name,self.Y_train.parquet_file_name,self.wt_train.parquet_file_name]]
    elif dataset == "test":
      parquet_files = [[self.X_test.parquet_file_name,self.Y_test.parquet_file_name,self.wt_test.parquet_file_name]]

    # Get simulated data
    sim_dp = DataProcessor(
      parquet_files,
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : self.data_parameters
      }
    )
    sim = sim_dp.GetFull("dataset", functions_to_apply=["untransform"])
    sim.loc[:, "y"] = 0.0

    # Get synthetic data
    synth_dp = DataProcessor(
      [[partial(self.Sample, sim.loc[:,self.data_parameters["Y_columns"]])]],
      "generator",
      n_events = len(sim),
      options = {
        "parameters" : self.data_parameters,
        "batch_size" : len(sim)
      }
    )
    synth = synth_dp.GetFull("dataset", functions_to_apply=["untransform"])
    synth.loc[:,self.data_parameters["Y_columns"]] = sim.loc[:,self.data_parameters["Y_columns"]]
    synth.loc[:, "wt"] = 1.0
    synth.loc[:, "y"] = 1.0

    # Get unique Y and equalise weights
    Y_unique = np.unique(sim.loc[:, self.data_parameters["Y_columns"]].to_numpy())
    for y in Y_unique:
      inds = (synth.loc[:,self.data_parameters["Y_columns"]].to_numpy() == y).flatten()
      sum_sim = np.sum(sim.loc[inds, "wt"])
      sum_synth = np.sum(synth.loc[inds, "wt"])
      synth.loc[inds, "wt"] *= sum_sim/sum_synth

    total = pd.concat([synth, sim], ignore_index=True)
    del sim, synth

    # Make training and testing datasets
    X_wt_train, X_wt_test, y_train, y_test = train_test_split(total.loc[:, self.data_parameters["X_columns"] + self.data_parameters["Y_columns"] + ["wt"]], total.loc[:,"y"], test_size=0.5, random_state=42)
    wt_train = X_wt_train.loc[:,"wt"].to_numpy()
    wt_test = X_wt_test.loc[:,"wt"].to_numpy()
    X_train = X_wt_train.loc[:,self.data_parameters["X_columns"] + self.data_parameters["Y_columns"]].to_numpy()
    X_test = X_wt_test.loc[:,self.data_parameters["X_columns"] + self.data_parameters["Y_columns"]].to_numpy()
    del X_wt_train, X_wt_test

    # Do fix if negative weights
    neg_train_wt_inds = (wt_train<0)
    neg_train_wt = (len(wt_train[neg_train_wt_inds]) > 0)
    if neg_train_wt:
      y_train[neg_train_wt_inds] += 2
      wt_train[neg_train_wt_inds] *= -1

    # Train separator
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train, sample_weight=wt_train)
    y_prob = clf.predict_proba(X_test)[:,1]

    # Get auc
    fpr, tpr, _ = roc_curve(y_test, y_prob, sample_weight=wt_test)
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    auc = roc_curve_auc(fpr, tpr)
    return float(abs(0.5-auc) + 0.5)

  def Probability(self, X, Y, return_log_prob=True):

    # Remove unneccessary Y components
    Y = Y.loc[:,self.data_parameters["Y_columns"]]

    # Set up Y correctly
    if len(Y) == 1:
      Y = pd.DataFrame(np.tile(Y.to_numpy().flatten(), (len(X), 1)), columns=Y.columns)
    elif len(Y) == 0:
      Y = pd.DataFrame(np.tile(np.array([]), (len(X), 1)), columns=Y.columns)

    # Transform Y input
    Y_dp = DataProcessor(
      [[Y]],
      "dataset",
      options = {
        "parameters" : self.data_parameters,
      }
    )
    if len(Y.columns) != 0:
      Y = Y_dp.GetFull(
        method="dataset",
        functions_to_apply = ["transform"]
      )

    # Transform X
    X_dp = DataProcessor(
      [[X]],
      "dataset",
      options = {
        "parameters" : self.data_parameters,
      }
    )
    X = X_dp.GetFull(
      method="dataset",
      functions_to_apply = ["transform"]
    )

    # Set up inputs for probability
    data = {
      "parameters" : X.loc[:,self.data_parameters["X_columns"]].to_numpy().astype(np.float32),
      "direct_conditions" : Y.to_numpy().astype(np.float32),
    }

    # Add zeros column onto 1d datasets - need to add integral aswell
    if self.fix_1d:
      data["parameters"] = np.column_stack((data["parameters"].flatten(), np.zeros(len(data["parameters"]))))

    # Get probability
    log_prob = pd.DataFrame(self.amortizer.log_posterior(data), columns=["log_prob"])

    # Untransform probability
    prob_dp = DataProcessor(
      [[log_prob]],
      "dataset",
      options = {
        "parameters" : self.data_parameters,
      }
    )
    if len(Y.columns) != 0:
      log_prob = prob_dp.GetFull(
        method="dataset",
        functions_to_apply = ["untransform"]
      ) 

    # return probability
    if return_log_prob:
      return log_prob.to_numpy()
    else:
      return np.exp(log_prob.to_numpy())