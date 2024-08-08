import copy
import os
import pickle
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from itertools import product
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split
from scipy.optimize import root_scalar

from data_processor import DataProcessor
from plotting import plot_histograms, plot_spline_and_thresholds
from useful_functions import GetYName, MakeDirectories, GetPOILoop, GetNuisanceLoop

class InputPlot():

  def __init__(self):
    """
    A class to preprocess the datasets and produce the data 
    parameters yaml file as well as the train, test and 
    validation datasets.
    """
    # Required input which is the location of a file
    self.cfg = None
    self.parameters = None

    # Other
    self.verbose = True
    self.data_input = "data/"
    self.plots_output = "plots/"

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
    # Open the config
    with open(self.cfg, 'r') as yaml_file:
      cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)    

    self.dataset_loop = ["train","test","val"] if cfg["preprocess"]["train_test_val_split"].count(":") == 2 else ["train", "val"]

    # Run plotting of POIs
    if self.verbose:
      print("- Making plots of POIs")
    for info in GetPOILoop(cfg, parameters):
      self._PlotX(info["poi"], info["freeze"], info["extra_name"], parameters)

    # Run plotting of nuisances
    if self.verbose:
      print("- Making plots of nuisances")
    for info in GetNuisanceLoop(cfg, parameters):
      self._PlotX(info["nuisance"], info["freeze"], info["extra_name"], parameters)

    # Run plotting of Y distributions
    if self.verbose:
      print("- Making plots of Y distributions")
    self._PlotY(parameters)

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
    inputs = [
      self.cfg,
      self.parameters,
    ]
    return inputs
        
  def _PlotX(self, vary, freeze, extra_name, parameters, n_bins=40):

    baseline_selection = " & ".join([f"({k}=={v})" for k,v in freeze.items()])
    for data_split in self.dataset_loop:
      dp = DataProcessor(
        [[f"{self.data_input}/X_{data_split}.parquet", f"{self.data_input}/Y_{data_split}.parquet", f"{self.data_input}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : None if baseline_selection == "" else baseline_selection,
          "parameters" : parameters
        }
      )
      for transform in [False, True]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]
        else:
          functions_to_apply = ["untransform","transform"]

        count = dp.GetFull(method="count", functions_to_apply=["untransform"])
        sumw = dp.GetFull(method="sum", functions_to_apply=["untransform"])
        if count == 0: continue
        unique_values = dp.GetFull(method="unique", functions_to_apply=["untransform"])
        for col in parameters["X_columns"]:
          hists = []
          hist_names = []
          bins = dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, functions_to_apply=functions_to_apply, column=col, ignore_discrete=True)
          if vary in unique_values.keys():
            for uc in sorted(unique_values[vary]):
              selection = f"({vary}=={uc})"
              hist, bins = dp.GetFull(method="histogram", bins=bins, functions_to_apply=functions_to_apply, extra_sel=selection, column=col)
              hists.append(hist)
              hist_names.append(GetYName([uc], purpose="plot", prefix="y="))
          else:
            hist, bins = dp.GetFull(method="histogram", bins=bins, functions_to_apply=functions_to_apply, column=col)
            hists.append(hist)
            hist_names.append(None)

          extra_name_for_plot = f"{extra_name}_{data_split}"
          if transform:
            extra_name_for_plot += "_transformed"

          plot_name = self.plots_output+f"/X_distributions_varying_{vary}_against_{col}{extra_name_for_plot}"
          plot_histograms(
            bins[:-1],
            hists,
            hist_names,
            title_right = "",
            name = plot_name,
            x_label = col,
            y_label = "Events",
            anchor_y_at_0 = True
          )

  def _PlotY(self, parameters, n_bins=40):

    for data_split in self.dataset_loop:
      dp = DataProcessor(
        [[f"{self.data_input}/Y_{data_split}.parquet", f"{self.data_input}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : None,
          "parameters" : parameters
        }
      )
      for transform in [False, True]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]

        for col in parameters["Y_columns"]:

          bins = dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, functions_to_apply=functions_to_apply, column=col, discrete_binning=False, ignore_discrete=True)
          hist, bins = dp.GetFull(method="histogram", bins=bins, functions_to_apply=functions_to_apply, column=col, discrete_binning=False)

          extra_name_for_plot = f"{data_split}"
          if transform:
            extra_name_for_plot += "_transformed"
          plot_name = self.plots_output+f"/Y_distributions_for_{col}_{extra_name_for_plot}"
          plot_histograms(
            bins[:-1],
            [hist],
            [None],
            title_right = "",
            name = plot_name,
            x_label = col,
            y_label = "Events",
            anchor_y_at_0 = True,
            drawstyle = "steps-mid",
          )