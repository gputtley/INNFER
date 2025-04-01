import copy
import os
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial

from data_processor import DataProcessor
from plotting import plot_histograms_with_ratio
from useful_functions import InitiateRegressionModel, LoadConfig, MakeDirectories

class PlotRegression():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None
    self.cfg = None

    # other
    self.data_input = "data/"
    self.plots_output = "plots/"
    self.model_name = None
    self.parameter = None
    self.verbose = True
    self.test_name = "test"

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

    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Make y from train and test
    loop = ["train"]
    if self.test_name is not None:
      loop.append(self.test_name)

    for data_split in loop:

      if self.verbose:
        print(f"- Building data processors for the {data_split} dataset")

      parameters_removed_standardisation = copy.deepcopy(parameters['regression'][self.parameter])
      if "standardisation" in parameters_removed_standardisation.keys():
        if "wt_shift" in parameters_removed_standardisation["standardisation"].keys():
          del parameters_removed_standardisation["standardisation"]["wt_shift"]

      pred_df = DataProcessor(
        [f"{self.data_input}/pred_{data_split}.parquet"] + [f"{parameters['regression'][self.parameter]['file_loc']}/{i}_{data_split}.parquet" for i in ["X","wt"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : parameters_removed_standardisation,
          "functions" : ["untransform"]
        }
      )


      y_df = DataProcessor(
        [[f"{parameters['regression'][self.parameter]['file_loc']}/{i}_{data_split}.parquet" for i in ["X","y","wt"]]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : parameters['regression'][self.parameter],
          "functions" : ["untransform"]
        }
      )

      # Make 1D histogram
      if self.verbose:
        print(f"- Making distributions of average weights")

      for col in parameters["regression"][self.parameter]["X_columns"]:

        # get nominal sums
        bins = y_df.GetFull(method="bins_with_equal_spacing", column=col, bins=20, ignore_quantile=0.02)
        nom_hist, _, _ = y_df.GetFull(method="histogram_and_uncert", column=col, bins=bins)

        # get shift applied histogram
        def change_weight(df):
          df.loc[:,"wt"] *= df.loc[:,"wt_shift"]
          return df
        y_hist, y_hist_uncerts, _ = y_df.GetFull(method="histogram_and_uncert", column=col, bins=bins, functions_to_apply=[change_weight])
        pred_hist, pred_hist_uncerts, _ = pred_df.GetFull(method="histogram_and_uncert", column=col, bins=bins, functions_to_apply=[change_weight])

        y_ave = y_hist/nom_hist
        y_ave_uncert = y_hist_uncerts/nom_hist
        pred_ave = pred_hist/nom_hist
        pred_ave_uncert = pred_hist_uncerts/nom_hist

        plot_histograms_with_ratio(
          [[y_ave, pred_ave]],
          [[y_ave_uncert, pred_ave_uncert]],
          [["True", "Regressed"]],
          bins,
          xlabel = col,
          ylabel="Average Output",
          name=f"{self.plots_output}/average_weight_{col}_{data_split}",      
          ratio_range = [0.9,1.1] 
        )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add plots
    for data_split in ["train", self.test_name]:
      for col in cfg["variables"]:
        outputs += [f"{self.plots_output}/average_weight_{col}_{data_split}.pdf"]
        
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initialise inputs
    inputs = []

    # Add parameters
    inputs += [self.parameters]

    # Add pred input
    inputs += [
      f"{self.data_input}/pred_train.parquet", 
      f"{self.data_input}/pred_{self.test_name}.parquet",
    ]
  
    # Add data input
    inputs += [f"{self.data_input}/{i}_{self.test_name}.parquet" for i in ["X","y","wt"]]
    inputs += [f"{self.data_input}/{i}_train.parquet" for i in ["X","y","wt"]]

    return inputs

        