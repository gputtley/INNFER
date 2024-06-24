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
    self.show2sigma = False

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
    other_results = {}
    for ind, info in enumerate(self.val_loop):
      info_name = GetYName(info["row"],purpose="plot",prefix="y=")
      for col in list(info["initial_best_fit_guess"].columns):
        if col in self.freeze.keys():
          continue
        with open(f"{self.data_input}/{self.file_name}_{col}_{ind}.yaml", 'r') as yaml_file:
          scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
        crossings = {k:v/scan_results_info["row"][scan_results_info["columns"].index(col)] for k,v in scan_results_info["crossings"].items()}
        if col not in results.keys():
          results[col] = {info_name:crossings}
        else:
          results[col][info_name] = crossings

        for other_key, other_val in self.other_input.items():
          if other_key not in other_results.keys():
            other_results[other_key] = {}
          with open(f"{other_val[0]}/{other_val[1]}_{col}_{ind}.yaml", 'r') as yaml_file:
            other_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader) 
          other_crossings = {k:v/other_results_info["row"][other_results_info["columns"].index(col)] for k,v in other_results_info["crossings"].items()}
          if col not in other_results[other_key].keys():
            other_results[other_key][col] = {info_name:other_crossings}
          else:
            other_results[other_key][col][info_name] = other_crossings

    plot_summary(
      results, 
      name = f"{self.plots_output}/summary{self.extra_plot_name}", 
      other_summaries = other_results,
      show2sigma = self.show2sigma,
      nominal_name = "" if len(list(other_results.keys())) == 0 else "Nominal"
    )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [
      f"{self.plots_output}/summary{self.extra_plot_name}.pdf"
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    for ind, info in enumerate(self.val_loop):
      info_name = GetYName(info["row"],purpose="plot",prefix="y=")
      for col in list(info["initial_best_fit_guess"].columns):
        if col in self.freeze.keys():
          continue
        inputs.append(f"{self.data_input}/{self.file_name}_{col}_{ind}.yaml")

        for other_key, other_val in self.other_input.items():
          inputs.append(f"{other_val[0]}/{other_val[1]}_{col}_{ind}.yaml")

    return inputs
