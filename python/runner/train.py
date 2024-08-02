import yaml

class Train():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None
    self.architecture = None

    # other
    self.use_wandb = False
    self.verbose = True
    self.disable_tqdm = False
    self.data_output = "data/"
    self.plots_output = "plots/"
    self.save_extra_name = ""
    self.test_name = "test"
    self.no_plot = False

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

    # Build model
    if self.verbose:
      print("- Building the model")
    from network import Network
    network = Network(
      f"{parameters['file_loc']}/X_train.parquet",
      f"{parameters['file_loc']}/Y_train.parquet", 
      f"{parameters['file_loc']}/wt_train.parquet", 
      f"{parameters['file_loc']}/X_{self.test_name}.parquet",
      f"{parameters['file_loc']}/Y_{self.test_name}.parquet", 
      f"{parameters['file_loc']}/wt_{self.test_name}.parquet",
      options = {
        **architecture,
        **{
          "plot_dir" : self.plots_output,
          "disable_tqdm" : self.disable_tqdm,
          "use_wandb" : self.use_wandb,
          "data_parameters" : parameters, # Do we actually need this to train?
        }
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
    network.Train()

    # Saving model
    if self.verbose:
      print("- Saving the model and its architecture")
    network.Save(name=f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}.h5")
    with open(f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}_architecture.yaml", 'w') as file:
      yaml.dump(architecture, file)

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    outputs = [
      f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}.h5",
      f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}_architecture.yaml",
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    inputs = [
      self.parameters,
      self.architecture,
      f"{parameters['file_loc']}/X_train.parquet",
      f"{parameters['file_loc']}/Y_train.parquet", 
      f"{parameters['file_loc']}/wt_train.parquet", 
      f"{parameters['file_loc']}/X_{self.test_name}.parquet",
      f"{parameters['file_loc']}/Y_{self.test_name}.parquet", 
      f"{parameters['file_loc']}/wt_{self.test_name}.parquet",
    ]
    return inputs

        