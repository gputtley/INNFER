import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import bayesflow as bf
import tensorflow as tf
from data_loader import DataLoader
from innfer_trainer import InnferTrainer

class Network():

  def __init__(self, X_train, Y_train, wt_train, X_test, Y_test, wt_test, options={}):

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

    self._SetOptions(options)

    # Model and trainer store
    self.inference_network = None
    self.amortizer = None
    self.trainer = None
    self.lr_scheduler = None
    
    # Data parquet files
    self.big_batch_size = 100000
    self.X_train = DataLoader(X_train, batch_size=self.batch_size)
    self.Y_train = DataLoader(Y_train, batch_size=self.batch_size)
    self.wt_train = DataLoader(wt_train, batch_size=self.batch_size)
    self.X_test = DataLoader(X_test, batch_size=self.big_batch_size)
    self.Y_test = DataLoader(Y_test, batch_size=self.big_batch_size)
    self.wt_test = DataLoader(wt_test, batch_size=self.big_batch_size)


  def _SetOptions(self, options):

    for key, value in options.items():
      setattr(self, key, value)

  def BuildModel(self):

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

    self.inference_net.save_weights(name)

  def Load(self, name="model.h5"):

    self.BuildModel()
    _ = self.inference_net(self.X_train.LoadNextBatch().to_numpy(),self.Y_train.LoadNextBatch().to_numpy())
    self.inference_net.load_weights(name)

  def Train(self, name="loss.pdf"):

    history = self.trainer.train_innfer(
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
    )

    #if self.plot_loss:
    #  f = bf.diagnostics.plot_losses(history["train_losses"], history["val_losses"])
    #  f.savefig(f"{self.plot_dir}/{name}")