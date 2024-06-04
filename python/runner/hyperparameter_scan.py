import yaml
import wandb
from random_word import RandomWords

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
    self.wandb_project_name = "innfer"
    self.wandb_submit_name = "innfer"
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
    # Setup wandb
    if self.use_wandb:
      with open(self.architecture, 'r') as yaml_file:
        architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
      r = RandomWords()
      wandb.init(project=self.wandb_project_name, name=f"{self.wandb_submit_name}_{r.get_random_word()}", config=architecture)

    # Train models
    t = self._SetupTrain()
    t.Run()

    # Get performance metrics
    pf = self._SetupPerformanceMetrics()
    pf.Run()

    # Write performance metrics to wandb
    if self.use_wandb:
      metric_name = f"{self.data_output}/metrics{self.save_extra_name}.yaml"
      with open(metric_name, 'r') as yaml_file:
        metric = yaml.load(yaml_file, Loader=yaml.FullLoader)
      wandb.log(metric)
      wandb.finish()



  def Outputs(self):
    """
    Return a list of outputs given by class
    """

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
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
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
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    pf = PerformanceMetrics()
    pf.Configure(
      {
        "model" : f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}.h5",
        "architecture" : f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}_architecture.yaml",
        "parameters" : self.parameters,
        "data_output" : self.data_output,
        "save_extra_name" : self.save_extra_name
      }
    )
    return pf