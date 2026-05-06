import copy
import os
import yaml

import numpy as np
import pandas as pd

from functools import partial

from plotting import plot_stacked_histogram_with_ratio, plot_stacked_unrolled_2d_histogram_with_ratio
from python.worker.useful_functions import GetDefaultsInModel
from useful_functions import GetYName, Translate, GetDictionaryEntry

class BinnedDistributions():

  def __init__(self):

    self.parameters = None
    self.categories = None
    self.val_info = None
    self.constraints = None
    self.binned_fit_input = None
    self.data_input_keys = None
    self.plots_output = None
    self.poi = None
    self.verbose = False
    self.extra_plot_name = ""
    self.inference_options = None
    self.include_uncertainty = False
    self.extra_hypotheses = []
    self.ratio_range = [0.5, 1.5]
    self.extra_inputs = []
    self.binned_observed_from_predicted = False

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    # Make singular inputs as dictionaries
    if isinstance(self.parameters, str):
      with open(self.parameters, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
      self.parameters = {parameters['file_name'] : self.parameters}

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"

  def Run(self):

    # Build yields
    from infer import Infer
    infer_class = Infer()
    infer_class.Configure(
      {
        "parameters" : {self.category : self.parameters},
        "binned_fit_morph_col" : self.poi,
        "binned_data_input_parameters_key" : {self.category : self.data_input_keys},
        "inference_options" : self.inference_options,

      }
    )
    yields = infer_class._BuildBinYields()[self.category]
    Y = pd.DataFrame({k: [v] for k, v in self.val_info.items()})

    # Evaluate the bins values at the hypotheses points
    stack_hists = {}
    for k, v in yields.items():
      stack_hists[k] = np.array(v(Y))

    # Get the data histogram
    if not self.binned_observed_from_predicted:
      data_hist = None
      for k, v in self.data_input_keys.items():
        with open(self.parameters[k], 'r') as yaml_file:
          parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
        entry = GetDictionaryEntry(parameters, v)
        if data_hist is None:
          data_hist = np.array(entry)
        else:
          data_hist += np.array(entry)
    else:
      data_hist = np.zeros_like(stack_hists[list(stack_hists.keys())[0]])
      for k, v in yields.items():
        data_hist += np.array(v(Y))

    # Get the uncertainty on the stack histograms from the constraints if required
    stack_uncertainty_up = np.zeros_like(stack_hists[list(stack_hists.keys())[0]])
    stack_uncertainty_down = np.zeros_like(stack_hists[list(stack_hists.keys())[0]])
    sum_stack_hists = np.sum(list(stack_hists.values()), axis=0)
    if self.include_uncertainty:
      for k, v in self.constraints.items():
        v_Y_up = pd.DataFrame({k1: [v1] for k1, v1 in v["up"].items()})
        v_Y_down = pd.DataFrame({k1: [v1] for k1, v1 in v["down"].items()})
        total_shifted_hist_up = np.zeros_like(sum_stack_hists)
        total_shifted_hist_down = np.zeros_like(sum_stack_hists)
        for k, v in yields.items():
          total_shifted_hist_up += np.array(v(v_Y_up))
          total_shifted_hist_down += np.array(v(v_Y_down))
        stack_uncertainty_up += (np.maximum(0, total_shifted_hist_up - sum_stack_hists))**2
        stack_uncertainty_down += (np.maximum(0, sum_stack_hists - total_shifted_hist_down))**2
      stack_uncertainty_up = np.sqrt(stack_uncertainty_up)
      stack_uncertainty_down = np.sqrt(stack_uncertainty_down)

    # Get extra hypotheses if required
    extra_hypotheses = {}
    for extra_hypothesis in self.extra_hypotheses:
      extra_hypothesis_Y = copy.deepcopy(Y)
      extra_hypothesis_name = []
      for k, v in extra_hypothesis.items():
        extra_hypothesis_Y[k] = [float(v)]
        extra_hypothesis_name += [f"{Translate(k)}={v}"]
      extra_hypothesis_name = ", ".join(extra_hypothesis_name)
      extra_hypothesis_hist = np.zeros_like(sum_stack_hists)
      for k, v in yields.items():
        extra_hypothesis_hist += np.array(v(extra_hypothesis_Y))
      extra_hypotheses[extra_hypothesis_name] = extra_hypothesis_hist

    # Get the axis text
    axis_text = Translate(self.category)
    varied_columns = []
    for extra_hypothesis in self.extra_hypotheses:
      for k in extra_hypothesis.keys():
        if k not in varied_columns:
          varied_columns += [k]
    if len(varied_columns) > 0:
      axis_text += "\n"
      for k in varied_columns:
        axis_text += f"{Translate(k)}={Y[k][0]}, "
      axis_text = axis_text[:-2]

    # Running plotting functions
    if self.verbose:
      print(f"- Making binned distribution plots")

    plot_stacked_histogram_with_ratio(
      data_hist, 
      {Translate(k): v for k, v in stack_hists.items()},
      self.binned_fit_input["binning"], 
      data_name="Data", 
      xlabel=Translate(self.binned_fit_input["variable"]),
      ylabel="Events",
      name=f"{self.plots_output}/binned_distribution_category_{self.category}{self.extra_plot_name}", 
      data_errors=np.sqrt(data_hist), 
      stack_hist_errors_asym = {"down": stack_uncertainty_down, "up": stack_uncertainty_up},
      axis_text=axis_text,
      use_stat_err=False,
      extra_hists=extra_hypotheses,
      draw_ratio=True,
      ratio_range=self.ratio_range,
      )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    outputs += [f"{self.plots_output}/binned_distribution_category_{self.category}{self.extra_plot_name}.pdf"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for v in self.parameters.values():
      inputs += [v]

    inputs += self.extra_inputs

    return inputs
