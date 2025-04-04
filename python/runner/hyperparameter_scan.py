import wandb
import yaml

from random_word import RandomWords

from train_density import TrainDensity
from density_performance_metrics import DensityPerformanceMetrics

class HyperparameterScan():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.cfg = None
    self.parameters = None
    self.architecture = None

    # other
    self.file_name = None
    self.data_input = "data/"
    self.use_wandb = False
    self.wandb_project_name = "innfer"
    self.wandb_submit_name = "innfer"
    self.verbose = True
    self.disable_tqdm = False
    self.data_output = "data/"
    self.save_extra_name = ""
    self.density_performance_metrics = None
 

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

      if self.verbose:
        print("- Initialising wandb")

      with open(self.architecture, 'r') as yaml_file:
        architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
      r = RandomWords()
      wandb.init(project=self.wandb_project_name, name=f"{self.wandb_submit_name}_{r.get_random_word()}", config=architecture)

    # Train models
    if self.verbose:
      print("- Training the models")
    t = self._SetupTrain()
    t.Run()

    # Get performance metrics
    if self.verbose:
      print("- Getting the performance metrics")
    pf = self._SetupPerformanceMetrics()
    pf.Run()

    # Write performance metrics to wandb
    if self.use_wandb:
      if self.verbose:
        print("- Writing performance metrics to wandb")
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

    outputs = list(set(t.Outputs() + pf.Outputs()))
    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    t = self._SetupTrain()
    pf = self._SetupPerformanceMetrics()

    pf_inputs = pf.Inputs()
    t_outputs = t.Outputs()

    inputs = list(set(t.Inputs() + [i for i in pf_inputs if i not in t_outputs]))

    return inputs


  def _SetupTrain(self):

    t = TrainDensity()
    t.Configure(     
      {     
        "parameters" : self.parameters,
        "architecture" : self.architecture,
        "file_name" : self.file_name,
        "data_input" : f"{self.data_input}/{self.file_name}/density",
        "data_output" : self.data_output,
        "no_plot" : True,
        "disable_tqdm" : self.disable_tqdm,
        "use_wandb" : self.use_wandb,
        "initiate_wandb" : self.use_wandb,
        "wandb_project_name" : self.wandb_project_name,
        "wandb_submit_name" : self.wandb_submit_name,
        "save_extra_name" : self.save_extra_name,
        "verbose" : self.verbose,        
      }
    )

    return t

  def _SetupPerformanceMetrics(self):

    pf = DensityPerformanceMetrics()
    pf.Configure(
      {
        "cfg" : self.cfg,
        "file_name" : self.file_name,
        "parameters" : self.parameters,
        "model_input" : self.data_output,
        "data_input" : self.data_input,
        "data_output" : self.data_output,
        "do_inference": "inference" in self.density_performance_metrics,
        "do_loss": "loss" in self.density_performance_metrics,
        "do_histogram_metrics": "histogram" in self.density_performance_metrics,
        "do_multidimensional_dataset_metrics": "multidim" in self.density_performance_metrics,
        "save_extra_name": self.save_extra_name,
        "verbose" : self.verbose,     
      }
    )

    return pf