import copy
import os
import yaml

import numpy as np
import pandas as pd

from functools import partial

from plotting import plot_stacked_histogram_with_ratio, plot_stacked_unrolled_2d_histogram_with_ratio
from useful_functions import GetYName, Translate

class BinnedDistributions():

  def __init__(self):

    self.parameters = None

    self.Y_data = None
    self.Y_stack = None
    self.pois = None
    self.nuisances = None
    self.binned_fit_input = None

    self.yield_function = "default"
    self.plots_output = "plots/"
    self.verbose = True
    self.scale_to_yield = False
    self.extra_plot_name = ""
    self.other_input_files = []
    self.other_output_files = []
    self.data_type = "sim"
    self.data_file = None
    self.test_name = "test"

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

    from infer import Infer
    infer_class = Infer()
    infer_class.Configure(
      {
        "parameters" : self.parameters,
        "true_Y" : self.Y_data,
        "pois" : self.pois,
        "nuisances" : self.nuisances,
        "test_name" : self.test_name,
        "binned_fit_input" : self.binned_fit_input,
        "data_type" : self.data_type if self.data_type != "asimov" else "sim",
        "data_file" : self.data_file,
      }
    )

    infer_class.yields = infer_class._BuildYieldFunctions()
    categories = infer_class._BuildCategories()
    yields = infer_class._BuildBinYields()
    dps = infer_class._BuildDataProcessors()

    bin_start = 0
    for cat_ind, cat_info in categories.items():
      data_hist = None
      sim_hists = {}
      for file_name in dps.keys():
        hist, hist_uncert, _ = dps[file_name].GetFull(
          method = "histogram_and_uncert",
          extra_sel = cat_info[0],
          column = cat_info[1],
          bins = cat_info[2],
        )
        if data_hist is None:
          data_hist = copy.deepcopy(hist)
          data_hist_uncert = copy.deepcopy(hist_uncert)
        else:
          data_hist += hist
          data_hist_uncert = (data_hist_uncert**2 + hist_uncert**2)**0.5
        sim_hists[f"{file_name} {GetYName(self.Y_stack, purpose='plot', prefix='y=')}"] = np.array([yields[file_name][bin_num](self.Y_stack) for bin_num in range(bin_start, bin_start+len(cat_info[2])-1)])
      bin_start += len(cat_info[2])-1

      # Get names for plot
      if self.data_type in ["sim","asimov"]:
        data_plot_name = GetYName(self.Y_data, purpose="plot", prefix="Asimov y=")
        data_hist = None
        data_hist_uncert = None
        draw_ratio = False
      else:
        data_plot_name = "Data"
        data_hist = np.array(data_hist)
        data_hist_uncert = np.array(data_hist_uncert)
        draw_ratio = True
      sample_plot_name = GetYName(self.Y_stack, purpose="plot", prefix="Asimov y=")

      # Running plotting functions
      if self.verbose:
        print(f"- Making binned distribution plots")

      plot_stacked_histogram_with_ratio(
        data_hist, 
        sim_hists, 
        cat_info[2], 
        data_name=data_plot_name, 
        xlabel=Translate(cat_info[1]),
        ylabel="Events",
        name=f"{self.plots_output}/binned_distribution_category{cat_ind}{self.extra_plot_name}", 
        data_errors=data_hist_uncert, 
        stack_hist_errors=np.zeros(len(sim_hists[list(sim_hists.keys())[0]])), 
        axis_text=f"Category {cat_ind}",
        use_stat_err=False,
        draw_ratio=draw_ratio,
        )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for file_name in self.parameters.keys():
      with open(self.parameters[file_name], 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
      inputs += [
        self.parameters[file_name],
        f"{parameters['file_loc']}/X_full.parquet",
        f"{parameters['file_loc']}/Y_full.parquet", 
        f"{parameters['file_loc']}/wt_full.parquet", 
      ]

    # Add inputs from the dataset being used
    if self.data_type == "data":
      inputs += [self.data_file]

    # Add other outputs
    inputs += self.other_input_files

    return inputs
