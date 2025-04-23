import yaml

import numpy as np

from plotting import plot_summary_per_val
from useful_functions import Translate

class SummaryPerVal():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.val_info = None
    self.val_ind = None

    self.file_name = "scan_results"
    self.data_input = "data"
    self.nominal_name = "Nominal"
    self.plots_output = "plots"
    self.other_input = {}
    self.freeze = {}
    self.extra_plot_name = ""
    self.show2sigma = False
    self.verbose = True
    self.column_loop = []
    self.constraints = []

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

    # Open results
    if self.verbose:
      print("- Loading in results")

    results_with_constraints = {}
    results_without_constraints = {}
    truth_without_constraints = {}
    names = {}

    if self.val_info is not None:
      plot_text = ", ".join([f"{Translate(k)}={round(v,2)}" for k, v in self.val_info.items()])

    # Find rows that are changing
    for col in self.column_loop:

      translated_col = Translate(col)
      if col in self.freeze.keys():
        continue
      with open(f"{self.data_input}/{self.file_name}_{col}_{self.val_ind}.yaml", 'r') as yaml_file:
        scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      if col in self.constraints:
        results_with_constraints[translated_col] = [{k:v-scan_results_info["row"][scan_results_info["columns"].index(col)] for k,v in scan_results_info["crossings"].items()}]
      else:
        results_without_constraints[translated_col] =  [scan_results_info["crossings"]]
        if "row" in scan_results_info.keys():
          truth_without_constraints[translated_col] = scan_results_info["row"][scan_results_info["columns"].index(scan_results_info["varied_column"])]
        else:
          truth_without_constraints[translated_col] = None
      names[translated_col] = [self.nominal_name]

      for other_key, other_val in self.other_input.items():
        with open(f"{other_val[0]}/{other_val[1]}_{col}_{self.val_ind}.yaml", 'r') as yaml_file:
          other_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

        if col in self.constraints:
          results_with_constraints[translated_col] += [{k:v-other_results_info["row"][other_results_info["columns"].index(col)] for k,v in other_results_info["crossings"].items()}]
        else:
          results_without_constraints[translated_col] += [other_results_info["crossings"]]
        names[translated_col] += [other_key]

    if self.verbose:
      print("- Plotting the summary")

    plot_summary_per_val(
      results_with_constraints,
      results_without_constraints,
      truth_without_constraints,
      names,
      show2sigma = self.show2sigma,
      plot_text = plot_text,
      plot_name = f"{self.plots_output}/summary_per_val_{self.val_ind}{self.extra_plot_name}",
    )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [f"{self.plots_output}/summary_per_val_{self.val_ind}{self.extra_plot_name}.pdf"]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initiate inputs
    inputs = []

    # Add the input files
    for col in self.column_loop:
      if col in self.freeze.keys():
        continue
      inputs += [f"{self.data_input}/{self.file_name}_{col}_{self.val_ind}.yaml"]
      for other_key, other_val in self.other_input.items():
        inputs += [f"{other_val[0]}/{other_val[1]}_{col}_{self.val_ind}.yaml"]

    return inputs
