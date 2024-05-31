import yaml

from train import Train
from performance_metrics import PerformanceMetrics

class HyperparameterScan():

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
    self.save_extra_name = ""

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
    # Train models
    t = self._SetupTrain()
    t.Run()

    pf = self._SetupPerformanceMetrics()
    pf.Run()


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    t = self._SetupTrain()
    pf = self._SetupPerformanceMetrics()

    outputs = t.Output() + pf.Outputs()
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
      f"{parameters['file_loc']}/X_test.parquet",
      f"{parameters['file_loc']}/Y_test.parquet", 
      f"{parameters['file_loc']}/wt_test.parquet",
    ]
    return inputs

  def _SetupTrain(self):
    t = Train()
    t.Configure(
      {
        "parameters" : self.parameters,
        "architecture" : self.architecture,
        "data_output" : self.data_output,
        "use_wandb" : self.use_wandb,
        "disable_tqdm" : self.disable_tqdm,
        "no_plot" : True,
        "save_extra_name" : self.save_extra_name
      }
    )
    return t

  def _SetupPerformanceMetrics(self):
    pf = PerformanceMetrics()
    pf.Configure(
      {
        "model" : f"{self.data_output}/{file_name}{self.save_extra_name}.h5",
        "architecture" : f"{self.data_output}/{file_name}{self.save_extra_name}_architecture.yaml",
        "parameters" : self.parameters,
        "data_output" : self.data_output,
        "save_extra_name" : self.save_extra_name
      }
    )
    return pf