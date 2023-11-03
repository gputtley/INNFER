import bayesflow as bf
import tensorflow as tf
from data_processor import DataProcessor
from innfer_trainer import InnferTrainer

class Model():

  def __init__(self, data_processor, options={}):

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
      "decay_steps" : 1000,
    } 

    # Model and trainer store
    self.inference_network = None
    self.amortizer = None
    self.trainer = None
    self.lr_scheduler = None
    
    # Data
    self.data_processor = data_processor

    # Running parameters
    self.plot_loss = True
    self.plot_dir = "plots"

    self._SetOptions(options)

  def _SetOptions(self, options):

    for key, value in options.items():
      setattr(self, key, value)

  def BuildModel(self):

    if self.data_processor.data["train"]["X"].shape[1] == 1 and self.coupling_design in ["spline","interleaved"]:
      self.data_processor.add_dummy_X = True
      latent_dim = 2
    else:
      latent_dim = self.data_processor.data["train"]["X"].shape[1]

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

    self.trainer = bf.trainers.Trainer(amortizer=self.amortizer, configurator=config, default_lr=self.learning_rate, memory=False)

    if self.lr_scheduler_name == "ExponentialDecay":
      self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        self.trainer.default_lr, 
        decay_steps=self.lr_scheduler_options["decay_steps"], 
        decay_rate=self.lr_scheduler_options["decay_rate"]
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
    data = self.data_processor.PreProcess()
    _ = self.inference_net(data["train"]["X"][:2,:],data["train"]["Y"][:2,:])
    self.inference_net.load_weights(name)

  def Train(self, name="loss.pdf"):

    data = self.data_processor.PreProcess(purpose="training") 

    history = self.trainer.train_offline(
      data["train"], 
      epochs=self.epochs, 
      batch_size=self.batch_size, 
      validation_sims=data["test"] if "test" in data.keys() else None, 
      early_stopping=self.early_stopping,
      optimizer=self.optimizer,
    )

    if self.plot_loss:
      if "test" in data.keys():
        f = bf.diagnostics.plot_losses(history["train_losses"], history["val_losses"])
      else:
        f = bf.diagnostics.plot_losses(history["train_losses"])
      f.savefig(f"{self.plot_dir}/{name}")
