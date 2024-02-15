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
    # Model parameters
    self.coupling_design = "affine"
    self.units_per_coupling_layer = 128
    self.num_dense_layers = 2
    self.activation = "relu"

    # Training parameters
    self.dropout = True
    self.mc_dropout = True
    self.early_stopping = True
    self.epochs = 15
    self.batch_size = 2**6
    self.learning_rate = 1e-3
    self.permutation = "learnable"
    self.optimizer_name = "Adam" 
    self.lr_scheduler_name = "ExponentialDecay"
    self.lr_scheduler_options = {
      "decay_rate" : 0.5,
    } 

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

    settings = {
      "dense_args": dict(
        kernel_regularizer=None, 
        units=self.units_per_coupling_layer, 
        activation=self.activation), 
      "dropout": self.dropout,
      "mc_dropout": self.mc_dropout,
      "num_dense": self.num_dense_layers,
      }

    if self.coupling_design == "interleaved":
      settings = {
        "affine" : settings,
        "spline" : settings
      }

    self.inference_net = bf.networks.InvertibleNetwork(
      num_params=latent_dim,
      num_coupling_layers=self.num_coupling_layers,
      permutation=self.permutation,
      coupling_design=self.coupling_design,
      coupling_settings=settings
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

  def Load(self, name="model.h5"):
    """
    Load the model weights.

    Args:
        name (str): Name of the file containing the model weights.
    """
    self.BuildModel()
    _ = self.inference_net(self.X_train.LoadNextBatch().to_numpy(), self.Y_train.LoadNextBatch().to_numpy())
    self.X_train.batch_num = 0
    self.Y_train.batch_num = 0
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
      fix_1d_spline=self.fix_1d_spline,
      disable_tqdm=self.disable_tqdm,
      use_wandb=self.use_wandb
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

  def Sample(self, Y, columns=None, n_events=10**6, transform=False, Y_transformed=False):
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
    if columns is not None:
      column_indices = [columns.index(col) for col in self.data_parameters["Y_columns"]]
      Y = Y[:,column_indices]
    pp = PreProcess()
    pp.parameters = self.data_parameters
    if not Y_transformed:
      Y = pp.TransformData(pd.DataFrame(Y, columns=self.data_parameters["Y_columns"])).to_numpy()
    data = {
      "direct_conditions" : Y.astype(np.float32)
    }
    synth = self.amortizer.sample(data, 1)[:,0,:]
    if self.fix_1d_spline:
      synth = synth[:,0].reshape(-1,1)

    if not transform:
      synth = pp.UnTransformData(pd.DataFrame(synth, columns=self.data_parameters["X_columns"])).to_numpy()
    
    return synth

  def Probability(self, X, Y, y_columns=None, seed=42, change_zero_prob=True, return_log_prob=False, run_normalise=True, elements_per_batch=10**6):
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
    Y_initial = copy.deepcopy(Y)
    if len(Y) == 1: Y = np.tile(Y, (len(X), 1))

    Y_name = GetYName(Y_initial.flatten(), purpose="file", round_to=18)

    if Y_name in self.probability_store.keys():

      log_prob = self.probability_store[Y_name]

    else:

      pp = PreProcess()
      pp.parameters = self.data_parameters
      X_untransformed = copy.deepcopy(X)
      X = pp.TransformData(pd.DataFrame(X, columns=self.X_train.columns)).to_numpy()
      Y = pp.TransformData(pd.DataFrame(Y, columns=self.Y_train.columns)).to_numpy()
      data = {
        "parameters" : X.astype(np.float32),
        "direct_conditions" : Y.astype(np.float32),
      }
      del X, Y
      gc.collect()

      if self.fix_1d_spline:
        data["parameters"] = np.column_stack((data["parameters"].flatten(), np.zeros(len(data["parameters"]))))

      # Get probabilities
      tf.random.set_seed(seed)
      tf.keras.utils.set_random_seed(seed)

      # Get the probability in batches so not to encounter memory problems
      log_prob = np.array([])
      #batch_size = int(elements_per_batch/(data["parameters"].shape[1] + data["direct_conditions"].shape[1]))
      batch_size = int(elements_per_batch)
      n_batches = int(np.ceil(len(data["parameters"])/batch_size))
      for i in range(n_batches):
        batch_data = {}
        for k, v in data.items():
          batch_data[k] = v[i*batch_size:min((i+1)*batch_size,len(data["parameters"]))]
        log_prob = np.append(log_prob, self.amortizer.log_posterior(batch_data))
      #log_prob = self.amortizer.log_posterior(data)


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
    synth = self.Sample(Y, columns=y_columns, n_events=n_samples)

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

      _, edges = np.histogramdd(synth, bins=n_integral_bins)
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

  def SeparateDistributions(self, dataset="train"):
    """
    Gets the AUC score when trying to separate the datasets with a Boosted Decision Tree (BDT).

    Args:
        dataset (str): The dataset to use, either "train" or "test" (default is "train").

    Returns:
        float: Absolute difference between 0.5 and the AUC score.
    """
    print(">> Getting AUC score when trying to separate the datasets with a BDT.")

    if dataset == "train":
      Y = self.Y_train.LoadFullDataset()
      x1 = self.X_train.LoadFullDataset()
      wt1 = self.wt_train.LoadFullDataset()
    elif dataset == "test":
      Y = self.Y_test.LoadFullDataset()
      x1 = self.X_test.LoadFullDataset()
      wt1 = self.wt_test.LoadFullDataset()

    x1 = np.hstack((x1, Y))
    y1 = np.zeros(len(x1)).reshape(-1,1)
    wt1 = wt1.to_numpy()

    x2 = self.Sample(Y, transform=True, Y_transformed=True)
    x2 = np.hstack((x2, Y))
    y2 = np.ones(len(x2)).reshape(-1,1)
    wt2 = np.ones(len(x2)).reshape(-1,1)
    wt1 *= np.sum(wt2)/np.sum(wt1)

    x = np.vstack((x1,x2))
    y = np.vstack((y1,y2))
    wt = np.vstack((wt1,wt2))

    X_wt_train, X_wt_test, y_train, y_test = train_test_split(np.hstack((x,wt)), y, test_size=0.5, random_state=42)

    X_train = X_wt_train[:,:-1]
    X_test = X_wt_test[:,:-1]
    wt_train = X_wt_train[:,-1]
    wt_test = X_wt_test[:,-1]

    del x1, x2, y1, y2, wt1, wt2, Y, x, y, X_wt_train, X_wt_test
    gc.collect()

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train, sample_weight=wt_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob, sample_weight=wt_test)

    return float(abs(0.5-auc))