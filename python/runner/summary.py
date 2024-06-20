import yaml
import numpy as np

from useful_functions import MakeDirectories, GetYName
from plotting import plot_summary

class Summary():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.columns = None
    self.val_loop = None

    self.file_name = "scan_results"
    self.data_input = "data"
    self.plots_output = "plots"    
    self.other_input = {}
    self.freeze = {}
    self.extra_plot_name = ""

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    # Open bootstraps
    results = {}
    for ind, info in enumerate(self.val_loop):
      info_name = GetYName(info["row"],purpose="plot",prefix="y=")
      for col in list(info["initial_best_fit_guess"].columns):
        if col in self.freeze.keys():
          continue
        with open(f"{self.data_input}/{self.file_name}_{col}_{ind}.yaml", 'r') as yaml_file:
          scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

        crossings = {k:v/scan_results_info["row"][scan_results_info["columns"].index(col)]  for k,v in scan_results_info["crossings"].items()}
        if col not in results.keys():
          results[col] = {info_name:crossings}
        else:
          results[col][info_name] = crossings

    plot_summary(
      results, 
      name=f"{self.plots_output}/summary", 
    )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [
      f"{self.plots_output}/bootstrap_distribution_{self.column}{self.extra_file_name}.pdf"
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [
      f"{self.data_input}/bootstrap_results_{self.column}{self.extra_file_name}.yaml"
    ]
    return inputs
