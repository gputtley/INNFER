import yaml

import numpy as np

from plotting import plot_summary
from useful_functions import Translate

class Summary():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.val_loop = None

    self.file_name = "scan_results"
    self.data_input = "data"
    self.nominal_name = "Nominal"
    self.plots_output = "plots"
    self.chi_squared = None
    self.other_input = {}
    self.freeze = {}
    self.extra_plot_name = ""
    self.show2sigma = False
    self.verbose = True
    self.column_loop = []
    self.subtract = False

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
    results = {}
    other_results = {}
    for ind, info in enumerate(self.val_loop):

      # Find rows that are changing
      for col in self.column_loop:

        name = ", ".join([f"{Translate(k)}={v}" for k, v in info.items()])

        translated_col = Translate(col)
        if col in self.freeze.keys():
          continue
        with open(f"{self.data_input}/{self.file_name}_{col}_{ind}.yaml", 'r') as yaml_file:
          scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
        if not self.subtract:
          crossings = {k:v/scan_results_info["row"][scan_results_info["columns"].index(col)] for k,v in scan_results_info["crossings"].items()}
        else:
          crossings = {k:v-scan_results_info["row"][scan_results_info["columns"].index(col)] for k,v in scan_results_info["crossings"].items()}
        if translated_col not in results.keys():
          results[translated_col] = {name:crossings}
        else:
          results[translated_col][name] = crossings

        for other_key, other_val in self.other_input.items():
          if other_key not in other_results.keys():
            other_results[other_key] = {}
          with open(f"{other_val[0]}/{other_val[1]}_{col}_{ind}.yaml", 'r') as yaml_file:
            other_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
          if not self.subtract:
            other_crossings = {k:v/other_results_info["row"][other_results_info["columns"].index(col)] for k,v in other_results_info["crossings"].items()}
          else:
            other_crossings = {k:v-other_results_info["row"][other_results_info["columns"].index(col)] for k,v in other_results_info["crossings"].items()}
          if translated_col not in other_results[other_key].keys():
            other_results[other_key][translated_col] = {name:other_crossings}
          else:
            other_results[other_key][translated_col][name] = other_crossings

    if self.verbose:
      print("- Plotting the summary")
      
    plot_summary(
      results, 
      name = f"{self.plots_output}/summary{self.extra_plot_name}", 
      other_summaries = other_results,
      show2sigma = self.show2sigma,
      subtract = self.subtract,
      nominal_name = "" if len(list(other_results.keys())) == 0 else self.nominal_name,
      text = None if self.chi_squared is None else {col: r'$\chi^2/N_{dof}$ = ' + str(round(self.chi_squared[col]["all"],2)) for col in list(info["initial_best_fit_guess"].columns)},
      #y_label = f"Truth ({', '.join([Translate(i) for i in non_repeated_columns])})",
      y_label = "",
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
      for col in self.column_loop:
        if col in self.freeze.keys():
          continue
        inputs.append(f"{self.data_input}/{self.file_name}_{col}_{ind}.yaml")

        for other_key, other_val in self.other_input.items():
          inputs.append(f"{other_val[0]}/{other_val[1]}_{col}_{ind}.yaml")

    return inputs
