import time

import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import bayesflow as bf
import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader import DataLoader

from data_processor import DataProcessor
from innfer_trainer import InnferTrainer
from optimizer import Optimizer
from plotting import plot_histograms
from useful_functions import MakeDirectories

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  print("INFO: Using GPUs for BayesFlowNetwork")
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class BayesFlowNetwork():
  """
  Network class for building and training a BayesFlow normalising flow.
  """
  def __init__(self, X_train=None, Y_train=None, wt_train=None, X_test=None, Y_test=None, wt_test=None, options={}):
    """
    Network class for building and training BayesFlow neural networks.

    Parameters
    ----------
    X_train : str
        Path to the training data for features.
    Y_train : str
        Path to the training data for target variables.
    wt_train : str
        Path to the training data for weights.
    X_test : str
        Path to the test data for features.
    Y_test : str
        Path to the test data for target variables.
    wt_test : str
        Path to the test data for weights.
    options : dict, optional
        Additional options for customization (default is an empty dictionary).
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
    self.affine_mc_dropout = False
    self.affine_dropout_prob = 0.05

    # spline parameters
    self.spline_units_per_dense_layer = 128
    self.spline_num_dense_layers = 2
    self.spline_activation = "relu"
    self.spline_dropout = True
    self.spline_mc_dropout = False
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
    self.gradient_clipping_norm = None

    # Other
    self.disable_tqdm = False
    self.use_wandb = False
    self.save_model_per_epoch = False

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
    if X_train is not None:
      self.X_train = DataLoader(X_train, batch_size=self.batch_size)
    else:
      self.X_train = None
    if Y_train is not None:
      self.Y_train = DataLoader(Y_train, batch_size=self.batch_size)
    else:
      self.Y_train = None
    if wt_train is not None:
      self.wt_train = DataLoader(wt_train, batch_size=self.batch_size)
    else:
      self.wt_train = None
    if X_test is not None:
      self.X_test = DataLoader(X_test, batch_size=self.batch_size)
    else:
      self.X_test = None
    if Y_test is not None:
      self.Y_test = DataLoader(Y_test, batch_size=self.batch_size)
    else:
      self.Y_test = None
    if wt_test is not None:
      self.wt_test = DataLoader(wt_test, batch_size=self.batch_size)
    else:
      self.wt_test = None

    # Other
    self.fix_1d = (self.X_train.num_columns == 1)
    self.adaptive_lr_scheduler = None
    self.prob_integral_store = None
    self.prob_integral_store_Y = None
    self.prob_integral_batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    self.graph_mode = True
    if not self.graph_mode:
      tf.config.optimizer.set_jit(True)
    self.length_batch = None

    self._compute_log_prob = None


  def _SetOptions(self, options):
    """
    Set options for the Network instance.

    Parameters
    ----------
    options : dict
        Dictionary of options to set.
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

    optim = Optimizer()
    self.optimizer, self.lr_scheduler, self.adaptive_lr_scheduler = optim.GetOptimizer(
      self.X_train.num_rows,
      self.batch_size,
      self.epochs,
      optimizer_name=self.optimizer_name, 
      lr_scheduler_name=self.lr_scheduler_name,
      lr_scheduler_options=self.lr_scheduler_options,
      default_lr=self.learning_rate,
      gradient_clipping_norm=self.gradient_clipping_norm,
    )


  def Load(self, name="model.h5", seed=42):
    """
    Load the model weights from a specified file.

    Parameters
    ----------
    name : str, optional
        Name of the file containing the model weights. Default is 'model.h5'.
    seed : int, optional
        Seed value for reproducibility. Default is 42.
    """

    self.BuildModel()
    X_train_batch = self.X_train.LoadNextBatch().to_numpy()
    Y_train_batch = self.Y_train.LoadNextBatch().to_numpy()
    self.X_train.batch_num = 0
    self.Y_train.batch_num = 0
    _ = self.inference_net(X_train_batch, Y_train_batch)
    self.inference_net.load_weights(name)

  def Loss(self, X, Y, wt):
    self.BuildTrainer()
    loss = float(self.trainer._get_epoch_loss(DataLoader(X, batch_size=self.batch_size), DataLoader(Y, batch_size=self.batch_size), DataLoader(wt, batch_size=self.batch_size), 0))
    return loss

  @tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32), tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
  def _ComputeLogProb(self, parameters, direct_conditions):
    z, log_det_J = self.amortizer.inference_net.forward(parameters, direct_conditions)
    log_prob = self.amortizer.latent_dist.log_prob(z) + log_det_J
    return log_prob

  @tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32), tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
  def _ComputeGradient(self, parameters, direct_conditions):
    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
      tape.watch(direct_conditions)
      z, log_det_J = self.amortizer.inference_net.forward(parameters, direct_conditions)
      predictions = self.amortizer.latent_dist.log_prob(z) + log_det_J
    grad = tape.gradient(predictions, direct_conditions)
    return predictions, grad


  def compute_log_prob(self, parameters, direct_conditions):
      if self._compute_log_prob is None:
          # Get the shape of the first batch
          batch_size, n_params = parameters.shape
          _, n_conditions = direct_conditions.shape

          # Create the tf.function with fixed shape
          @tf.function(input_signature=[
              tf.TensorSpec(shape=[batch_size, n_params], dtype=tf.float32),
              tf.TensorSpec(shape=[batch_size, n_conditions], dtype=tf.float32)
          ], reduce_retracing=True)
          def _inner(parameters_tensor, conditions_tensor):
              z, log_det_J = self.amortizer.inference_net.forward(parameters_tensor, conditions_tensor)
              log_prob = self.amortizer.latent_dist.log_prob(z) + log_det_J
              return log_prob
          
          self._compute_log_prob = _inner

      return self._compute_log_prob(parameters, direct_conditions)


  def Probability(self, X, Y, return_log_prob=True, transform_X=True, transform_Y=True, no_fix=False, order=0, column_1=None, column_2=None, grad_of="direct_conditions"):
    """
    Calculate the logarithmic or exponential probability density function.

    Parameters
    ----------
    X : pandas.DataFrame
        Input data for features.
    Y : pandas.DataFrame
        Input data for target variables.
    return_log_prob : bool, optional
        Whether to return the logarithmic probability (True) or the exponential probability (False). 
        Default is True.
    transform_X : bool, optional
        Whether to transform the input data X. Default is True.
    transform_Y : bool, optional
        Whether to transform the input data Y. Default is True.
    no_fix : bool, optional
        If True, skip fixing 1D probability by ensuring integral is 1. Default is False.

    Returns
    -------
    numpy.ndarray
        Logarithmic or exponential probability density values.
    """

    # Remove unneccessary Y components
    X = X[self.data_parameters["X_columns"]]
    Y = Y[self.data_parameters["Y_columns"]]

    length_batch = len(X)
    if self.length_batch is None:
      self.length_batch = length_batch

    if self.fix_1d and not no_fix and len(self.data_parameters["X_columns"]) == 1:
      Y_initial = copy.deepcopy(Y)

    # Prepare datasets
    Y = self.PrepareY(X, Y, transform_Y=transform_Y)
    X = self.PrepareX(X, transform_X=transform_X)

    # Set up inputs for probability
    data = {
      "parameters" : X.to_numpy(np.float32).astype(np.float32),
      "direct_conditions" : Y.to_numpy(np.float32).astype(np.float32),
    }


    # Check type of gradient
    if isinstance(order,int):
      gradients = [order]
    else:
      gradients = order
    if 2 in gradients and column_1 is None and column_2 is None:
      raise ValueError("You must specifiy the columns required for the second derivative.")
    if isinstance(column_1, str):
      column_1 = [column_1]
    if isinstance(column_2, str):
      column_2 = [column_2]  
    if 2 in gradients and not (len(column_1)==1 and len(column_2)==1):
      raise ValueError("The second derivative must be done one column at a time.")

    # Conversion dict
    conversion = {
      "direct_conditions" : "Y_columns",
      "parameters" : "X_columns"
    }

    # Check columns
    if column_1 is None:
      column_1 = self.data_parameters[conversion[grad_of]]
    if column_2 is None:
      column_2 = self.data_parameters[conversion[grad_of]]

    # Get indices
    indices_1 = [self.data_parameters[conversion[grad_of]].index(col) for col in column_1 if col in self.data_parameters[conversion[grad_of]]]
    indices_2 = [self.data_parameters[conversion[grad_of]].index(col) for col in column_2 if col in self.data_parameters[conversion[grad_of]]]
    column_to_index_1 = {col : indices_1.index(self.data_parameters[conversion[grad_of]].index(col)) for col in column_1 if col in self.data_parameters[conversion[grad_of]]}

    # Add zeros column onto 1d datasets - need to add integral as well
    if self.fix_1d:
      data["parameters"] = np.column_stack((data["parameters"].flatten(), np.zeros(len(data["parameters"]))))


    # Get the log probs and gradients
    if order == 0 or order == [0]: # Get the log probability

      if self.graph_mode:
        data["parameters"] = tf.convert_to_tensor(tf.cast(data["parameters"], dtype=tf.float32), dtype=tf.float32)
        data["direct_conditions"] = tf.convert_to_tensor(tf.cast(data["direct_conditions"], dtype=tf.float32), dtype=tf.float32)

      #print("Tracing count before step:", self._ComputeLogProb.experimental_get_tracing_count())
      #if self.graph_mode and length_batch == self.length_batch:
      if self.graph_mode:
        log_probs_model = self._ComputeLogProb(data["parameters"], data["direct_conditions"])
        log_probs = [pd.DataFrame(log_probs_model.numpy(), columns=["log_prob"], dtype=np.float64)]
      #if self.graph_mode and length_batch == self.length_batch:
      #  #print("GM")
      #  #log_probs_model = self._ComputeLogProb(data["parameters"], data["direct_conditions"])
      #  log_probs_model = self.compute_log_prob(data["parameters"], data["direct_conditions"])
      #  log_probs = [pd.DataFrame(log_probs_model.numpy(), columns=["log_prob"], dtype=np.float64)]
      else:
        #print("NGM")
        log_probs = [pd.DataFrame(self.amortizer.log_posterior(data), columns=["log_prob"], dtype=np.float64)]
      #print("Tracing count after step:", self._ComputeLogProb.experimental_get_tracing_count())

    elif order == 1 or order == [1] or order == [0,1]: # Get the first derivative of the log probability
      
      if self.graph_mode:
        data["parameters"] = tf.convert_to_tensor(tf.cast(data["parameters"], dtype=tf.float32), dtype=tf.float32)
        data["direct_conditions"] = tf.convert_to_tensor(tf.cast(data["direct_conditions"], dtype=tf.float32), dtype=tf.float32)

      if len(indices_1) == 0:

        predictions = tf.convert_to_tensor(self.amortizer.log_posterior(data), dtype=tf.float32)
        first_derivative = tf.zeros((len(data["parameters"]), 1), dtype=tf.float32)

      else:

        if self.graph_mode:
          predictions, grad = self._ComputeGradient(data["parameters"], data["direct_conditions"])
        else:
          tf.keras.backend.clear_session()
          if not self.graph_mode:
            data["parameters"] = tf.convert_to_tensor(data["parameters"], dtype=tf.float32)
            data["direct_conditions"] = tf.convert_to_tensor(data["direct_conditions"], dtype=tf.float32)
          with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(data[grad_of])
            z, log_det_J = self.amortizer.inference_net.forward(data["parameters"], data["direct_conditions"])
            predictions = tf.reshape(self.amortizer.latent_dist.log_prob(z) + log_det_J, (-1, 1))
          grad = tape.gradient(predictions, data[grad_of])
        first_derivative = tf.gather(grad, indices_1, axis=1)
      
      # Make log_probs array
      log_probs = []
      if order == [0,1]:
        log_probs += [pd.DataFrame(predictions.numpy(), columns=["log_prob"], dtype=np.float64)]

      if order == [0,1] or order == 1 or order == [1]:
        first_derivative_for_all_columns = np.zeros((len(data["parameters"]),len(column_1)))
        for ind, col in enumerate(column_1):
          if col not in self.data_parameters[conversion[grad_of]]: continue
          first_derivative_for_all_columns[:, ind] = first_derivative.numpy()[:, column_to_index_1[col]]
        log_probs += [pd.DataFrame(first_derivative_for_all_columns, columns=[f"d_log_prob_by_d_{col}" for col in column_1], dtype=np.float64)]

    elif order == 2 or order == [2] or order == [1,2] or order == [0,1,2] or order == [0,2]: # Get the second derivative of the log probability

      # Get the second derivative
      data["parameters"] = tf.convert_to_tensor(data["parameters"], dtype=tf.float32)
      data["direct_conditions"] = tf.convert_to_tensor(data["direct_conditions"], dtype=tf.float32)

      if len(indices_1) == 0:

        predictions = tf.convert_to_tensor(self.amortizer.log_posterior(data), dtype=tf.float32)
        first_derivative = tf.zeros((len(data["parameters"]), 1), dtype=tf.float32)
        second_derivative = tf.zeros((len(data["parameters"]), 1), dtype=tf.float32)

      elif len(indices_2) == 0:

        if self.graph_mode:
          predictions, grad = self._ComputeGradient(data["parameters"], data["direct_conditions"])
        else:
          tf.keras.backend.clear_session()
          if not self.graph_mode:
            data["parameters"] = tf.convert_to_tensor(data["parameters"], dtype=tf.float32)
            data["direct_conditions"] = tf.convert_to_tensor(data["direct_conditions"], dtype=tf.float32)
          with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(data[grad_of])
            z, log_det_J = self.amortizer.inference_net.forward(data["parameters"], data["direct_conditions"])
            predictions = tf.reshape(self.amortizer.latent_dist.log_prob(z) + log_det_J, (-1, 1))
          grad = tape.gradient(predictions, data[grad_of])

        first_derivative = tf.gather(grad, indices_1, axis=1)
        second_derivative = tf.zeros((len(data["parameters"]), 1), dtype=tf.float32)
      
      else:

        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_2:
          tape_2.watch(data[grad_of])
          with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_1:
            tape_1.watch(data[grad_of])
            z, log_det_J = self.amortizer.inference_net.forward(data["parameters"], data["direct_conditions"])
            predictions = self.amortizer.latent_dist.log_prob(z) + log_det_J
          grad = tape_1.gradient(predictions, data[grad_of])
          first_derivative = tf.gather(grad, indices_1, axis=1)   

        grad_of_grad = tape_2.gradient(first_derivative, data[grad_of])
        second_derivative = tf.gather(grad_of_grad, indices_2, axis=1)
      
      # Make log_probs array
      log_probs = []
      if order == [0,1,2] or order == [0,2]:
        log_probs += [pd.DataFrame(predictions.numpy(), columns=["log_prob"], dtype=np.float64)]

      if order == [0,1,2] or order == [1,2]:
        first_derivative_for_all_columns = np.zeros((len(data["parameters"]),len(column_1)))
        for ind, col in enumerate(column_1):
          if col not in self.data_parameters[conversion[grad_of]]: continue
          first_derivative_for_all_columns[:, ind] = first_derivative.numpy()[:, column_to_index_1[col]]
        log_probs += [pd.DataFrame(first_derivative_for_all_columns, columns=[f"d_log_prob_by_d_{col}" for col in column_1], dtype=np.float64)]

      if order == 2 or order == [2] or order == [0,1,2] or order == [0,2] or order == [1,2]:
        log_probs += [pd.DataFrame(second_derivative.numpy(), columns=[f"d2_log_prob_by_d_{col1}_and_{col2}" for col1 in column_1 for col2 in column_2], dtype=np.float64)]

    # Untransform probabilities
    for ind in range(len(log_probs)):

      columns = list(log_probs[ind].columns)

      prob_dp = DataProcessor(
        [[log_probs[ind]]],
        "dataset",
        options = {
          "parameters" : self.data_parameters,
        }
      )
      log_probs[ind] = prob_dp.GetFull(
        method="dataset",
        functions_to_apply = ["untransform"] if transform_X else []
      ).loc[:, columns]

      # Fix 1d probability by ensuring integral is 1
      if self.fix_1d and not no_fix and order[ind] == 0:
        log_probs[ind] = log_probs[ind] - np.log(self.ProbabilityIntegral(Y_initial, verbose=True))
        log_probs[ind] = log_probs[ind] - np.log(1/(2*np.pi)**0.5)

      # return probability - change this for derivatives
      if return_log_prob:
        if log_probs[ind] is not None:
          log_probs[ind] = log_probs[ind].to_numpy()
        else:
          log_probs[ind] = np.zeros((len(X),1))
      else:
        log_probs[ind] = np.exp(log_probs[ind].to_numpy())

    if isinstance(order, list):
      return log_probs
    else:
      return log_probs[0]
    

  def ProbabilityIntegral(self, Y, n_integral_bins=10**6, n_samples=10**4, ignore_quantile=0.0, extra_fraction=0.5, seed=42, verbose=False):
    """
    Compute the integral of the probability density function over a specified range.

    Parameters
    ----------
    Y : array-like
        Values of the model parameters for which the probability integral is computed.
    n_integral_bins : int, optional
        Number of bins used for the integral calculation. Default is 10**6.
    n_samples : int, optional
        Number of samples used for generating synthetic data. Default is 10**4.
    ignore_quantile : float, optional
        Fraction of extreme values to ignore when computing the integral. Default is 0.0.
    extra_fraction : float, optional
        Fractional increase in the integration range around the median value. Default is 0.5.
    seed : int, optional
        Seed for random number generation. Default is 42.
    verbose : bool, optional
        Whether to print the computed integral value. Default is False.

    Returns
    -------
    float
        Computed integral of the probability density function.
    """

    # If in the store use that
    if self.prob_integral_store is not None:
      if Y.equals(self.prob_integral_store_Y):
        return self.prob_integral_store

    # Calculate probability
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    synth = self.Sample(Y, n_events=n_samples).to_numpy()
    n_integral_bins = int(n_integral_bins**(1/synth.shape[1]))
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

    # Do probs in batches
    num_batches = int(np.ceil(len(unique_values)/self.prob_integral_batch_size))
    for batch_num in range(num_batches):
      start_ind = batch_num*self.prob_integral_batch_size
      end_ind = min((batch_num+1)*self.prob_integral_batch_size, len(unique_values))
      if batch_num == 0:
        probs = self.Probability(pd.DataFrame(unique_values[start_ind:end_ind,:], columns=self.data_parameters["X_columns"], dtype=np.float64), Y, return_log_prob=False, transform_X=True, no_fix=True)
      else:
        probs = np.vstack((probs, self.Probability(pd.DataFrame(unique_values[start_ind:end_ind,:], columns=self.data_parameters["X_columns"], dtype=np.float64), Y, return_log_prob=False, transform_X=True, no_fix=True)))
    bin_volumes = np.prod(np.diff(edges)[:,0], axis=None, dtype=np.float128)
    integral = np.sum(probs, dtype=np.float128) * bin_volumes

    # Save to store
    self.prob_integral_store_Y = copy.deepcopy(Y)
    self.prob_integral_store = copy.deepcopy(integral)

    if verbose: 
      print(f"Integral for Y is {integral}")  

    return integral


  def PrepareX(self, X, transform_X=True):

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
      functions_to_apply = ["transform"] if transform_X else []
    )
    return X[self.data_parameters["X_columns"]]


  def PrepareY(self, X, Y, transform_Y=True):

    # Set up Y correctly
    if len(Y) == 1:
      Y = pd.DataFrame(np.tile(Y.to_numpy().flatten(), (len(X), 1)), columns=Y.columns, dtype=np.float64)
    elif len(Y) == 0:
      Y = pd.DataFrame(np.tile(np.array([]), (len(X), 1)), columns=Y.columns, dtype=np.float64)

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
        functions_to_apply = ["transform"] if transform_Y else []
      )
    return Y[self.data_parameters["Y_columns"]]


  def Sample(self, Y, n_events):
    """
    Generate synthetic data samples based on given conditions.

    Parameters
    ----------
    Y : DataFrame
        DataFrame containing values of the model parameters for which samples are generated.
    n_events : int
        Number of synthetic data samples to generate.

    Returns
    -------
    DataFrame
        DataFrame containing the synthetic data samples.
    """

    # Remove unneccessary Y components
    Y = Y.loc[:,self.data_parameters["Y_columns"]]

    # Set up Y correctly
    if len(Y) == 1:
      Y = pd.DataFrame(np.tile(Y.to_numpy().flatten(), (n_events, 1)), columns=Y.columns, dtype=np.float64)
    elif len(Y) == 0:
      Y = pd.DataFrame(np.tile(np.array([]), (n_events, 1)), columns=Y.columns, dtype=np.float64)

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

    if not self.fix_1d:
      synth_df = pd.DataFrame(synth, columns=self.data_parameters["X_columns"])
    else:
      synth_df = pd.DataFrame(synth[:,0], columns=self.data_parameters["X_columns"])
    total_nans = synth_df.isna().sum().sum()
    rows_with_nan = synth_df[synth_df.isna().any(axis=1)]
    if total_nans > 0:
      synth[synth_df.isna().any(axis=1)] = self.amortizer.sample({"direct_conditions" : Y[synth_df.isna().any(axis=1)].to_numpy().astype(np.float32)}, 1)

    # Fix 1d couplings
    if self.fix_1d:
      synth = synth[:,0].reshape(-1,1)

    # Untransform the dataset
    synth_dp = DataProcessor(
      [[pd.DataFrame(synth, columns=self.data_parameters["X_columns"], dtype=np.float64)]],
      "dataset",
      options = {
        "parameters" : self.data_parameters,
      }
    )
    synth  = synth_dp.GetFull(
      method="dataset",
      functions_to_apply = ["untransform"]
    )

    return synth

  def Save(self, name="model.h5"):
    """
    Save the trained model weights.

    Parameters
    ----------
    name : str, optional
        Name of the file to save the model weights (default is "model.h5").
    """
    MakeDirectories(name)
    self.inference_net.save_weights(name)

  def Train(self, name="model.h5"):
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
      model_name=name,
      save_model_per_epoch=self.save_model_per_epoch,
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

      plot_histograms(
        range(1,len(self.trainer.loss_history._total_train_loss)),
        [self.trainer.loss_history._total_train_loss[1:], self.trainer.loss_history._total_val_loss[1:]],
        ["Train", "Test"],
        title_right = "",
        name = f"{self.plot_dir}/loss_no_zero",
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