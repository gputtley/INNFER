import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import bayesflow as bf
import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import DataLoader
from innfer_trainer import InnferTrainer
from preprocess import PreProcess
from plotting import plot_histograms

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
    self.num_coupling_layers = 8
    self.activation = "relu"

    # Training parameters
    self.dropout = True
    self.mc_dropout = True
    self.early_stopping = True
    self.epochs = 15
    self.batch_size = 2**6
    self.learning_rate = 1e-3, 
    self.permutation = "learnable",
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
    self.fix_1d_spline = ((self.X_train.num_columns == 1) and self.coupling_design in ["affine","interleaved"])

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

    self.trainer = InnferTrainer(amortizer=self.amortizer, configurator=config, default_lr=self.learning_rate, memory=False)

    if self.lr_scheduler_name == "ExponentialDecay":
      self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        self.trainer.default_lr, 
        decay_rate=self.lr_scheduler_options["decay_rate"],
        decay_steps=int(self.X_train.num_rows/self.batch_size)
      )
    else:
      print("ERROR: lr_schedule not valid.")

    if self.optimizer_name == "Adam":
      self.optimizer = tf.keras.optimizers.Adam(self.lr_scheduler)
    else:
      print("ERROR: optimizer not valid.")

  def Save(self, name="model.h5"):
    """
    Save the trained model weights.

    Args:
        name (str): Name of the file to save the model weights.
    """
    self.inference_net.save_weights(name)

  def Load(self, name="model.h5"):
    """
    Load the model weights.

    Args:
        name (str): Name of the file containing the model weights.
    """
    self.BuildModel()
    _ = self.inference_net(self.X_train.LoadNextBatch().to_numpy(),self.Y_train.LoadNextBatch().to_numpy())
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

  def Sample(self, Y, columns=None, n_events=10**5):
    """
    Generate synthetic data samples.

    Args:
        Y (np.ndarray): Input data for generating synthetic samples.
        columns (list): List of columns to consider from Y.
        n_events (int): Number of synthetic events to generate.

    Returns:
        np.ndarray: Synthetic data samples.
    """
    if columns is not None:
      column_indices = [columns.index(col) for col in self.Y_train.columns]
      Y = Y[:,column_indices]
    if len(Y) == 1: Y = np.tile(Y, (n_events, 1))
    pp = PreProcess()
    pp.parameters = self.data_parameters
    Y = pp.TransformData(pd.DataFrame(Y, columns=self.Y_train.columns)).to_numpy()
    data = {
      "direct_conditions" : Y.astype(np.float32)
    }
    synth = self.amortizer.sample(data, 1)[:,0,:]
    synth = pp.UnTransformData(pd.DataFrame(synth, columns=self.X_train.columns)).to_numpy()
    return synth

  def Probability(self, X, Y, y_columns=None, seed=42):
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
    if len(Y) == 1: Y = np.tile(Y, (len(X), 1))

    pp = PreProcess()
    pp.parameters = self.data_parameters
    X = pp.TransformData(pd.DataFrame(X, columns=self.X_train.columns)).to_numpy()
    Y = pp.TransformData(pd.DataFrame(Y, columns=self.Y_train.columns)).to_numpy()
    data = {
      "parameters" : X.astype(np.float32),
      "direct_conditions" : Y.astype(np.float32),
    }
    tf.random.set_seed(seed)
    prob = np.exp(self.model.amortizer.log_posterior(data))
    prob = pp.UnTransformProb(prob)
    return prob