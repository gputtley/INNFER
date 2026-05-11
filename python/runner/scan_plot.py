import yaml

import numpy as np

from scipy.interpolate import CubicSpline

from plotting import plot_likelihood
from useful_functions import Translate

class ScanPlot():

  def __init__(self):
    """
    A template class.
    """
    self.column = None

    self.data_input = "data"
    self.plots_output = "data"    
    self.extra_file_name = ""
    self.other_input = {}
    self.verbose = True
    self.extra_plot_name = ""
    self.val_info = None
    self.nominal_name = "Nominal"
    self.stat_syst_breakdown = False
    self.rezero_scan = False
    self.scan_colours = None
    self.scan_linestyles = None
    self.scan_no_result_text = False

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.extra_file_name != "":
      self.extra_file_name = f"_{self.extra_file_name}"

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    if self.verbose:
      print("- Loading in scan results")
    scan_results_file = f"{self.data_input}/scan_results_{self.column}{self.extra_file_name}.yaml"
    with open(scan_results_file, 'r') as yaml_file:
      scan_results = yaml.load(yaml_file, Loader=yaml.FullLoader)

    x = scan_results["scan_values"]
    if not self.rezero_scan:
      y = scan_results["nlls"]
      crossings = scan_results["crossings"]
    else:
      print(scan_results["crossings"])
      y, crossings = self._rezero_scan(scan_results["scan_values"], scan_results["nlls"], scan_results["crossings"])
      print(crossings)

    row = scan_results["row"]
    ind = scan_results["columns"].index(self.column)

    other_lklds = {}
    other_crossings = {}
    for key, val in self.other_input.items():
      other_scan_file = f"{val}/scan_results_{self.column}{self.extra_file_name}.yaml"
      with open(other_scan_file, 'r') as yaml_file:
        other_scan_results = yaml.load(yaml_file, Loader=yaml.FullLoader)
      if not self.rezero_scan:
        other_lklds[rf"{key}"] = [other_scan_results["scan_values"], other_scan_results["nlls"]]
        other_crossings[rf"{key}"] = other_scan_results["crossings"]
      else:
        rezerod_other = self._rezero_scan(other_scan_results["scan_values"], other_scan_results["nlls"], other_scan_results["crossings"])
        other_lklds[rf"{key}"] = [other_scan_results["scan_values"], rezerod_other[0]]
        other_crossings[rf"{key}"] = rezerod_other[1]

    if row is not None:
      plot_extra_name = ", ".join([f"{Translate(k)}={round(v,2)}" for k, v in self.val_info.items()])
    else:
      plot_extra_name = ""
    
    if self.verbose:
      print("- Plotting the likelihood scan")

    plot_likelihood(
      x, 
      y, 
      crossings, 
      name = f"{self.plots_output}/likelihood_scan_{self.column}{self.extra_file_name}{self.extra_plot_name}", 
      xlabel = Translate(self.column), 
      true_value = row[ind] if row is not None else None,
      under_result = f"Truth: {plot_extra_name}" if plot_extra_name != "" else "",
      label = None if len(list(other_lklds.keys())) == 0 else rf"{self.nominal_name}",
      other_lklds=other_lklds,
      other_crossings=other_crossings,
      stat_syst_breakdown=self.stat_syst_breakdown,
      colours = self.scan_colours,
      linestyles = self.scan_linestyles,
      no_result_text = self.scan_no_result_text
    )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [
      f"{self.plots_output}/likelihood_scan_{self.column}{self.extra_file_name}{self.extra_plot_name}.pdf"
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [
      f"{self.data_input}/scan_results_{self.column}{self.extra_file_name}.yaml"
    ]
    for key, val in self.other_input.items():
      inputs.append(f"{val}/scan_results_{self.column}{self.extra_file_name}.yaml")
    return inputs


  def _rezero_scan(self, x, y, crossings):
    spline = CubicSpline(x, y)
    x_linspace = np.linspace(x[0], x[-1], 1000)
    y_linspace = spline(x_linspace)
    min_y_ind = np.argmin(y_linspace)
    min_y = np.min(y_linspace)
    y = [y[i] - min_y for i in range(len(y))]
    y_linspace = [y_linspace[i] - min_y for i in range(len(y_linspace))]
    crossings = {0: x_linspace[min_y_ind]}
    for i in range(0, min_y_ind):
      min_bins = min(y_linspace[i], y_linspace[i+1])
      max_bins = max(y_linspace[i], y_linspace[i+1])
      if min_bins < 1 and max_bins >= 1:
        if -1 not in crossings:
          crossings[-1] = x_linspace[i] + (x_linspace[i+1] - x_linspace[i]) * (1 - y_linspace[i]) / (y_linspace[i+1] - y_linspace[i])
      if min_bins < 4 and max_bins >= 4:
        if -2 not in crossings:
          crossings[-2] = x_linspace[i] + (x_linspace[i+1] - x_linspace[i]) * (4 - y_linspace[i]) / (y_linspace[i+1] - y_linspace[i])
    for i in range(len(x_linspace)-2, min_y_ind, -1):
      min_bins = min(y_linspace[i], y_linspace[i+1])
      max_bins = max(y_linspace[i], y_linspace[i+1])
      if min_bins < 1 and max_bins >= 1:
        if 1 not in crossings:
          crossings[1] = x_linspace[i] + (x_linspace[i+1] - x_linspace[i]) * (1 - y_linspace[i]) / (y_linspace[i+1] - y_linspace[i])
      if min_bins < 4 and max_bins >= 4:
        if 2 not in crossings:
          crossings[2] = x_linspace[i] + (x_linspace[i+1] - x_linspace[i]) * (4 - y_linspace[i]) / (y_linspace[i+1] - y_linspace[i])
    return y, crossings