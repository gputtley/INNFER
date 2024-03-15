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
from preprocess import PreProcess
from plotting import plot_histograms
from scipy import integrate
from other_functions import GetYName, MakeDirectories
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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

    # Running parameters
    self.plot_loss = True
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
    self.fix_1d_spline = ((self.X_train.num_columns == 1) and self.coupling_design in ["spline","interleaved"])
    self.disable_tqdm = False
    self.use_wandb = False
    self.probability_store = {}
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
    if self.fix_1d_spline:
      print("WARNING: Running fix for 1D latent space for spline and interleaved couplings. Code may take longer to run.")
      latent_dim = 2

    # Print warning about mc_dropout:
    if self.affine_mc_dropout or self.spline_mc_dropout:
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

    self.amortizer = bf.amortizers.AmortizedPosterior(self.inference_net)

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
        outer_period=self.lr_scheduler_options["outer_period"]*int(self.X_train.num_rows/self.batch_size) if "outer_period" in self.lr_scheduler_options.keys() else self.epochs*int(self.X_train.num_rows/self.batch_size),
        inner_period=self.lr_scheduler_options["inner_period"]*int(self.X_train.num_rows/self.batch_size) if "inner_period" in self.lr_scheduler_options.keys() else 10*int(self.X_train.num_rows/self.batch_size),
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
        step_size=int(self.X_train.num_rows/self.batch_size)/self.lr_scheduler_options["cycles_per_epoch"] if "cycles_per_epoch" in self.lr_scheduler_options.keys() else int(self.X_train.num_rows/self.batch_size)/2,
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
    #tf.random.set_seed(seed)
    #tf.keras.utils.set_random_seed(seed)

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
      fix_1d_spline=self.fix_1d_spline,
      disable_tqdm=self.disable_tqdm,
      use_wandb=self.use_wandb,
      adaptive_lr_scheduler=self.adaptive_lr_scheduler,
      active_learning=self.active_learning,
      active_learning_options=self.active_learning_options
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

  def Sample(self, Y, columns=None, n_events=10**6, transform=False, Y_transformed=False, batch_size=10**6, seed=42):
    """
    Generate synthetic data samples.

    Args:
        Y (np.ndarray): Input data for generating synthetic samples.
        columns (list): List of columns to consider from Y.
        n_events (int): Number of synthetic events to generate.

    Returns:
        np.ndarray: Synthetic data samples.
    """
    Y = np.array(Y)
    if Y.ndim == 1 or len(Y) == 1: Y = np.tile(Y, (n_events, 1))
    
    if self.active_learning:
      Y = np.hstack((Y,np.zeros(len(Y)).reshape(-1,1)))

    if columns is not None:
      column_indices = [columns.index(col) for col in self.data_parameters["Y_columns"]]
      Y = Y[:,column_indices]
    pp = PreProcess()
    pp.parameters = self.data_parameters
    if not Y_transformed:
      Y = pp.TransformData(pd.DataFrame(Y, columns=self.data_parameters["Y_columns"], dtype=np.float64)).to_numpy()

    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    n_batches = int(np.ceil(len(Y)/batch_size))
    for i in range(n_batches):
      batch_data = {
        "direct_conditions" : Y[i*batch_size:min((i+1)*batch_size,len(Y))].astype(np.float32)
      }
      if i == 0:
        synth = self.amortizer.sample(batch_data, 1)[:,0,:]
      else:
        synth = np.vstack((synth, self.amortizer.sample(batch_data, 1)[:,0,:]))

    #print(synth)
    if self.fix_1d_spline:
      synth = synth[:,0].reshape(-1,1)

    synth_wt = np.ones(len(synth))

    if not transform:
      synth = pp.UnTransformData(pd.DataFrame(synth, columns=self.data_parameters["X_columns"], dtype=np.float64)).to_numpy()

    return synth, synth_wt

  def Probability(self, X, Y, y_columns=None, seed=2, change_zero_prob=True, return_log_prob=False, run_normalise=True, elements_per_batch=10**6):
    """
    Calculate probabilities for given data.

    Args:
        X (np.ndarray): Input data for features.
        Y (np.ndarray): Input data for target variables.
        y_columns (list): List of columns to consider from Y.
        seed (int): Seed for reproducibility.

    Returns:
        np.ndarray: Probabilities for the given data.
    """
    if y_columns is not None:
      column_indices = [y_columns.index(col) for col in self.Y_train.columns]
      Y = Y[:,column_indices]
    if self.active_learning:
      Y = np.hstack((Y,np.zeros(len(Y)).reshape(-1,1)))
    Y_initial = copy.deepcopy(Y)
    if len(Y) == 1: Y = np.tile(Y, (len(X), 1))

    Y_name = GetYName(Y_initial.flatten(), purpose="file", round_to=18)

    if Y_name in self.probability_store.keys():

      log_prob = self.probability_store[Y_name]

    else:

      pp = PreProcess()
      pp.parameters = self.data_parameters
      X_untransformed = copy.deepcopy(X)
      #tf.random.set_seed(seed)
      #tf.keras.utils.set_random_seed(seed)
      X = pp.TransformData(pd.DataFrame(X, columns=self.X_train.columns, dtype=np.float64)).to_numpy()
      Y = pp.TransformData(pd.DataFrame(Y, columns=self.Y_train.columns, dtype=np.float64)).to_numpy()
      data = {
        "parameters" : X.astype(np.float32),
        "direct_conditions" : Y.astype(np.float32),
      }
      del X, Y
      gc.collect()

      if self.fix_1d_spline:
        data["parameters"] = np.column_stack((data["parameters"].flatten(), np.zeros(len(data["parameters"]))))

      # Get probabilities
      #print(data["parameters"])
      #exit()

      # Get the probability in batches so not to encounter memory problems
      log_prob = np.array([])
      batch_size = int(elements_per_batch)
      n_batches = int(np.ceil(len(data["parameters"])/batch_size))
      for i in range(n_batches):
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        batch_data = {}
        for k, v in data.items():
          batch_data[k] = v[i*batch_size:min((i+1)*batch_size,len(data["parameters"]))]
          #print(k)
          #print(batch_data[k])
        #log_prob = self.amortizer.log_posterior(batch_data)
        #print("Prob")
        #print(log_prob[:10])
        log_prob = np.append(log_prob, self.amortizer.log_posterior(batch_data))

      log_prob = pp.UnTransformProb(log_prob, log_prob=True)

      if self.fix_1d_spline and run_normalise:
        log_prob -= np.log(self.ProbabilityIntegral(Y_initial, verbose=False))
        #log_prob += np.log(np.sqrt(2*np.pi)) # quicker but doesn't always work if you do not close the added dimension

      # Do checks
      if np.any(log_prob == -np.inf) and change_zero_prob:
        print("WARNING: Zero probabilities found.")
        indices = np.where(log_prob == -np.inf)
        print("Problem Row(s):")
        print(self.X_train.columns)
        print(X_untransformed[indices])
      if np.any(np.isnan(log_prob)):
        print("WARNING: NaN probabilities found.")
        indices = np.isnan(log_prob)
        log_prob[indices] = -np.inf
        print("Problem Row(s):")
        print(self.X_train.columns)
        print(X_untransformed[indices])

      # Change probs
      if change_zero_prob:
        indices = np.where(log_prob == -np.inf)
        log_prob[indices] = 1

      if len(self.probability_store.keys()) > 5:
        del self.probability_store[list(self.probability_store.keys())[0]]
      self.probability_store[Y_name] = copy.deepcopy(log_prob)

    if return_log_prob:
      return log_prob
    else:
      return np.exp(log_prob)
  
  def ProbabilityIntegral(self, Y, y_columns=None, n_integral_bins=10000, n_samples=10**4, ignore_quantile=0.000, extra_fraction=0.25, method="histogramdd", verbose=True):
    """
    Computes the integral of the probability density function over a specified range.

    Args:
        Y (array): The values of the model parameters for which the probability integral is computed.
        y_columns (list): List of column names corresponding to the model parameters (optional).
        n_integral_bins (int): Number of bins used for the integral calculation (default is 1000).
        n_samples (int): Number of samples used for generating synthetic data (default is 10**5).
        ignore_quantile (float): Fraction of extreme values to ignore when computing the integral (default is 0.0001).
        method (str): The method used for integration, either "histogramdd" or "scipy" (default is "histogramdd").
        verbose (bool): Whether to print the computed integral value (default is True).

    Returns:
        float: The computed integral of the probability density function.
    """
    synth, synth_wt = self.Sample(Y, columns=y_columns, n_events=n_samples)

    if method == "histogramdd":

      for col in range(synth.shape[1]):
        lower_value = np.quantile(synth[:,col], ignore_quantile)
        upper_value = np.quantile(synth[:,col], 1-ignore_quantile)
        if extra_fraction > 0.0:
          middle_value = np.quantile(synth[:,col], 0.5)
          lower_value = lower_value - (extra_fraction * (middle_value-lower_value))
          upper_value = upper_value + (extra_fraction * (upper_value-middle_value))
        trimmed_indices = ((synth[:,col] >= lower_value) & (synth[:,col] <= upper_value))
        synth = synth[trimmed_indices,:]

      _, edges = np.histogramdd(synth, weights=synth_wt, bins=n_integral_bins)
      bin_centers_per_dimension = [0.5 * (edges[dim][1:] + edges[dim][:-1]) for dim in range(len(edges))]
      meshgrid = np.meshgrid(*bin_centers_per_dimension, indexing='ij')
      unique_values = np.vstack([grid.flatten() for grid in meshgrid]).T

      probs = self.Probability(unique_values, Y, change_zero_prob=False, run_normalise=False)
      bin_volumes = np.prod(np.diff(edges)[:,0], axis=None)
      integral = np.sum(probs, dtype=np.float128) * bin_volumes

    elif method == "scipy":

      ranges = []
      for col in range(synth.shape[1]):
        ranges.append([np.quantile(synth[:,col], ignore_quantile),np.quantile(synth[:,col], 1-ignore_quantile)])

      def IntegrateFunc(*args):
        return self.Probability(np.array([list(args)]), Y, change_zero_prob=False)
      
      integral, _ = integrate.nquad(IntegrateFunc, ranges)

    if verbose: print(f"Integral for Y is {integral}")
    return integral

  def MakeSeparationDatasets(self, dataset="train", for_classification=False):
    """
    Prepares synthetic and simulated datasets for training or testing.

    Args:
        dataset (str): Dataset type ('train' or 'test'). Defaults to 'train'.
        for_classification (bool): Whether to prepare datasets for classification. Defaults to False.

    Returns:
        tuple: A tuple containing the synthetic and simulated datasets.
    """
    if dataset == "train":
      Y_sim = self.Y_train.LoadFullDataset().to_numpy()
      X_sim = self.X_train.LoadFullDataset().to_numpy()
      wt_sim = self.wt_train.LoadFullDataset().to_numpy()
    elif dataset == "test":
      Y_sim = self.Y_test.LoadFullDataset().to_numpy()
      X_sim = self.X_test.LoadFullDataset().to_numpy()
      wt_sim = self.wt_test.LoadFullDataset().to_numpy()

    Y_unique = np.unique(Y_sim)
    rng = np.random.RandomState(seed=42)
    Y_synth = rng.choice(Y_unique, size=len(Y_sim)).reshape(-1,1)
    X_synth, wt_synth = self.Sample(Y_synth, transform=True, Y_transformed=True)
    for y in Y_unique:
      sim_inds = (Y_sim == y).flatten()
      synth_inds = (Y_synth == y).flatten()
      wt_synth[synth_inds] *= np.sum(wt_sim[sim_inds])/np.sum(wt_synth[synth_inds])
      #print(f"Y = {y}, Sum Sim Wts = {float(np.sum(wt_sim[sim_inds]))}, Sum Synth Wts = {float(np.sum(wt_synth[synth_inds]))}")

    if not for_classification:
      return X_sim, Y_sim, wt_sim, X_synth, Y_synth, wt_synth 
    else:
      X = np.vstack((np.hstack((X_sim, Y_sim)),np.hstack((X_synth, Y_synth))))
      y = np.vstack((np.zeros(len(X_sim)).reshape(-1,1),np.ones(len(X_synth)).reshape(-1,1)))
      wt = np.vstack((wt_sim.reshape(-1,1), wt_synth.reshape(-1,1)))
      rng = np.random.RandomState(seed=42)
      indices = rng.permutation(X.shape[0])
      X = X[indices]
      y = y[indices]
      wt = wt[indices]
      return X, y, wt

  def auc(self, dataset="train"):
    """
    Gets the AUC score when trying to separate the datasets with a Boosted Decision Tree (BDT).

    Args:
        dataset (str): The dataset to use, either "train" or "test" (default is "train").

    Returns:
        float: Absolute difference between 0.5 and the AUC score.
    """
    print(">> Getting AUC score when trying to separate the datasets with a BDT")
    X, y, wt = self.MakeSeparationDatasets(dataset=dataset, for_classification=True)
    X_wt_train, X_wt_test, y_train, y_test = train_test_split(np.hstack((X,wt)), y, test_size=0.5, random_state=42)

    X_train = X_wt_train[:,:-1]
    X_test = X_wt_test[:,:-1]
    wt_train = X_wt_train[:,-1]
    wt_test = X_wt_test[:,-1]

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train, sample_weight=wt_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob, sample_weight=wt_test)
    return float(abs(0.5-auc))

  def r2(self, dataset="train"):
    """
    Calculate the R2 score for each feature in the specified dataset.

    Returns:
    - dict: A dictionary where keys are feature names and values are the R2 scores
            for those features.
    """
    """
    print(">> Getting R2")
    if dataset == "train":
      Y = self.Y_train.LoadFullDataset()
      x1 = self.X_train.LoadFullDataset()
      wt1 = self.wt_train.LoadFullDataset()
    elif dataset == "test":
      Y = self.Y_test.LoadFullDataset()
      x1 = self.X_test.LoadFullDataset()
      wt1 = self.wt_test.LoadFullDataset()

    x1 = x1.to_numpy()
    wt1 = wt1.to_numpy()

    Y_unique = np.unique(Y)
    Y_synth = np.random.choice(Y_unique, size=len(Y)).reshape(-1,1)
    x2, wt2 = self.Sample(Y_synth, transform=True, Y_transformed=True)    
    wt2 = wt2.reshape(-1,1)
    wt1 *= np.sum(wt2)/np.sum(wt1)

    """

    # get datasets
    x1, _, wt1, x2, _, wt2 = self.MakeSeparationDatasets(dataset="test")

    # remove negative weights
    if len(wt1[wt1 < 0]) > 0 or len(wt2[wt2 < 0]) > 0:
      ("WARNING: Ignoring negative weights")
      wt1[wt1 < 0] = 0
      wt2[wt2 < 0] = 0
    
    # normalise weights
    probs1 = wt1.flatten() / np.sum(wt1.flatten())
    probs2 = wt2.flatten() / np.sum(wt2.flatten())

    # apply the minimum size between x1 and x2
    size_min = np.min((len(x1), len(x2)))

    # resample data
    resampled_indices1 = np.random.choice(len(x1), size=size_min, p=probs1)
    resampled_indices2 = np.random.choice(len(x2), size=size_min, p=probs2)
    X1 = x1[resampled_indices1]
    X2 = x2[resampled_indices2]

    # convert to dataframes
    df_X1 = pd.DataFrame(X1)
    df_X2 = pd.DataFrame(X2)

    # calculate the R2 score for each column
    r2 = {}
    for i, col_name in enumerate(self.X_train.columns):
      X1_col = df_X1[i]
      X2_col = df_X2[i]
      X1_col = np.sort(X1_col)
      X2_col = np.sort(X2_col)
      X1_mean = X1_col.mean()
      r2_val = 1 - (np.sum((X1_col - X2_col)**2) / np.sum((X1_col - X1_mean)**2))
      r2[col_name] = float(r2_val)

    return r2

  def nrmse(self, dataset="train"):
    """
    Calculate the NRMSE score for each feature in the specified dataset.

    Returns:
    - dict: A dictionary where keys are feature names and values are the NRMSE scores
            for those features.
    """
    print(">> Getting NRMSE")

    # get datasets
    x1, _, wt1, x2, _, wt2 = self.MakeSeparationDatasets(dataset="test")

    # remove negative weights
    if len(wt1[wt1 < 0]) > 0 or len(wt2[wt2 < 0]) > 0:
      print("WARNING: Ignoring negative weights")
      wt1[wt1 < 0] = 0
      wt2[wt2 < 0] = 0
    
    # normalise weights
    probs1 = wt1.flatten() / np.sum(wt1.flatten())
    probs2 = wt2.flatten() / np.sum(wt2.flatten())

    # apply the minimum size between x1 and x2
    size_min = np.min((len(x1), len(x2)))

    # resample data
    resampled_indices1 = np.random.choice(len(x1), size=size_min, p=probs1)
    resampled_indices2 = np.random.choice(len(x2), size=size_min, p=probs2)
    X1 = x1[resampled_indices1]
    X2 = x2[resampled_indices2]

    # convert to dataframes
    df_X1 = pd.DataFrame(X1)
    df_X2 = pd.DataFrame(X2)

    # calculate the R2 score for each column
    nrmse = {}
    for i, col_name in enumerate(self.X_train.columns):
      X1_col = df_X1[i]
      X2_col = df_X2[i]
      X1_col = np.sort(X1_col)
      X2_col = np.sort(X2_col)
      rmse = np.sqrt(np.sum((X1_col - X2_col)**2)/len(X1_col))
      nrmse_val = rmse / (X1_col.max() - X1_col.min())
      nrmse[col_name] = float(nrmse_val)

    return nrmse

