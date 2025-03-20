import wandb
import yaml

from random_word import RandomWords

from useful_functions import InitiateDensityModel

class TrainDensity():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None
    self.architecture = None

    # other
    self.use_wandb = False
    self.initiate_wandb = False
    self.wandb_project_name = "innfer"
    self.wandb_submit_name = "innfer"
    self.verbose = True
    self.disable_tqdm = False
    self.data_output = "data/"
    self.plots_output = "plots/"
    self.save_extra_name = ""
    self.test_name = "test"
    self.no_plot = False
    self.save_model_per_epoch = False
    self.model_type = "BayesFlow"

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):
    """
    Run the code utilising the worker classes
    """
    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the architecture in
    if self.verbose:
      print("- Loading in the architecture")
    with open(self.architecture, 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if self.initiate_wandb:

      if self.verbose:
        print("- Initialising wandb")

      r = RandomWords()
      wandb.init(project=self.wandb_project_name, name=f"{self.wandb_submit_name}_{r.get_random_word()}", config=architecture)

    # Build model
    if self.verbose:
      print("- Building the model")
    network = InitiateDensityModel(
      architecture,
      parameters['density']['file_loc'],
      test_name = self.test_name,
      options = {
        "plot_dir" : self.plots_output,
        "disable_tqdm" : self.disable_tqdm,
        "use_wandb" : self.use_wandb,
        "data_parameters" : parameters["density"],
      }
    )

    network.BuildModel()
    
    if self.no_plot:
      network.plot_loss = False
      network.plot_lr = False

    # Training model
    if self.verbose:
      print("- Training the model")
    network.BuildTrainer()
    network.save_model_per_epoch = self.save_model_per_epoch
    network.Train(name=f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}.h5")

    # Saving model architecture
    if self.verbose:
      print("- Saving the model and its architecture")
    with open(f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}_architecture.yaml", 'w') as file:
      yaml.dump(architecture, file)

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Add model and architecture
    outputs = [
      f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}.h5",
      f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}_architecture.yaml",
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Set inputs
    inputs = [
      self.parameters,
      self.architecture,
      f"{parameters['density']['file_loc']}/X_train.parquet",
      f"{parameters['density']['file_loc']}/Y_train.parquet", 
      f"{parameters['density']['file_loc']}/wt_train.parquet", 
      f"{parameters['density']['file_loc']}/X_{self.test_name}.parquet",
      f"{parameters['density']['file_loc']}/Y_{self.test_name}.parquet", 
      f"{parameters['density']['file_loc']}/wt_{self.test_name}.parquet",
    ]

    return inputs

        