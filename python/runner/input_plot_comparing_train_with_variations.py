import os
import yaml

import numpy as np

from data_processor import DataProcessor
from plotting import plot_histograms_with_ratio, plot_learned_nuisance_variations
from useful_functions import LoadConfig, Translate

class InputPlotComparingTrainWithVariations():

  def __init__(self):
    """
    A class to compare training data with variations.
    """
    # Default values - these will be set by the configure function
    self.cfg = None
    self.open_cfg = None
    self.model_info = None
    self.file_name = None
    self.category = None
    self.train_data_input = None
    self.nominal_data_input = None
    self.up_data_input = None
    self.down_data_input = None
    self.plots_output = None
    self.nuisance = None
    self.sim_type = "val"
    self.ratio_range = [0.9,1.1]
    self.train_data_split = "train"
    self.verbose = True
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    self.half_window = 0.4
    
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
    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      if self.verbose:
        print("- Loading in config")
      cfg = LoadConfig(self.cfg)

    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.model_info['parameters'], 'r') as yaml_file:
      parameters = yaml.safe_load(yaml_file)
    parameter = self.model_info["parameter"]

    # Build data processor for train
    files = [f"{self.train_data_input}/{i}_{self.train_data_split}.parquet" for i in ["X","y","wt"]]
    train_dp = DataProcessor(
      [files],
      "parquet",
      batch_size = self.batch_size,
      wt_name = "wt",
      options = {
        "parameters" : parameters['classifier'][parameter],
      },
    )

    # Build validation data processor for variations
    nominal_files = [f"{self.nominal_data_input}/{i}_{self.sim_type}.parquet" for i in ["X","wt"]]
    up_files = [f"{self.up_data_input}/{i}_{self.sim_type}.parquet" for i in ["X","wt"]]
    down_files = [f"{self.down_data_input}/{i}_{self.sim_type}.parquet" for i in ["X","wt"]]

    nominal_dp = DataProcessor(
      [nominal_files],
      "parquet",
      batch_size = self.batch_size,
      wt_name = "wt",
      options = {
      },
    )
    up_dp = DataProcessor(
      [up_files],
      "parquet",
      batch_size = self.batch_size,
      wt_name = "wt",
      options = {
      },
    )
    down_dp = DataProcessor(
      [down_files],
      "parquet",
      batch_size = self.batch_size,
      wt_name = "wt",
      options = {
      },
    )

    # Loop over columns
    for col in cfg["variables"]:
      if self.verbose:
        print(f"- Plotting {col} for train and variations")
  
      train_nom_hist, train_nom_bins = train_dp.GetFull(
        method="histogram", 
        bins=40, 
        column=col, 
        functions_to_apply=["untransform","selection"],
        density=True,
        #extra_sel = f"(classifier_truth == 1) & ({parameter} > -{self.half_window}) & ({parameter} < {self.half_window})"
        extra_sel = f"(classifier_truth == 0)"
      )
      train_up_hist, _ = train_dp.GetFull(
        method="histogram", 
        bins=train_nom_bins, 
        column=col, 
        functions_to_apply=["untransform","selection"],
        density=True,
        extra_sel = f"(classifier_truth == 1) & ({parameter} > 1-{self.half_window}) & ({parameter} < 1+{self.half_window})"
      )
      train_down_hist, _ = train_dp.GetFull(
        method="histogram", 
        bins=train_nom_bins, 
        column=col, 
        functions_to_apply=["untransform","selection"],
        density=True,
        extra_sel = f"(classifier_truth == 1) & ({parameter} > -1-{self.half_window}) & ({parameter} < -1+{self.half_window})"
      )
      val_nom_hist, val_nom_uncert, _ = nominal_dp.GetFull(
        method="histogram_and_uncert", 
        bins=train_nom_bins, 
        column=col, 
        density=True,
      )
      val_up_hist, val_up_uncert, _ = up_dp.GetFull(
        method="histogram_and_uncert", 
        bins=train_nom_bins, 
        column=col, 
        density=True,
      )
      val_down_hist, val_down_uncert, _ = down_dp.GetFull(
        method="histogram_and_uncert", 
        bins=train_nom_bins, 
        column=col, 
        density=True,
      )

      plot_histograms_with_ratio(
        [[train_nom_hist, val_nom_hist], [train_up_hist, val_up_hist], [train_down_hist, val_down_hist]],
        [[np.zeros_like(train_nom_hist), np.zeros_like(val_nom_hist)], [np.zeros_like(train_up_hist), np.zeros_like(val_up_hist)], [np.zeros_like(train_down_hist), np.zeros_like(val_down_hist)]],
        [["Train Nominal", "Validation Nominal"], ["Train Up", "Validation Up"], ["Train Down", "Validation Down"]],
        train_nom_bins,
        xlabel = Translate(col),
        ylabel = "Density",
        name = f"{self.plots_output}/train_vs_variations_{col}_{self.nuisance}",
        ratio_range = self.ratio_range,
        anchor_y_at_0 = True,
      )

      plot_learned_nuisance_variations(
        train_nom_hist,
        train_up_hist,
        train_down_hist,
        val_nom_hist,
        val_up_hist,
        val_down_hist,
        val_nom_uncert,
        val_up_uncert,
        val_down_uncert,
        train_nom_bins,
        xlabel = Translate(col),
        output_name = f"{self.plots_output}/train_vs_variations_double_ratio_{col}_{self.nuisance}",
        ratio_range_line_vs_errorbar = self.ratio_range,
        ratio_range_variations = self.ratio_range,
        line_caption = "Train",
        errorbar_caption = "Validation",
      )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []

    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      cfg = LoadConfig(self.cfg)

    # Add the output files to the list
    for col in cfg["variables"]:
      outputs.append(f"{self.plots_output}/train_vs_variations_{col}_{self.nuisance}.pdf")
      outputs.append(f"{self.plots_output}/train_vs_variations_double_ratio_{col}_{self.nuisance}.pdf")

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    # Add config file
    inputs += [self.cfg]

    # Add parameters file
    inputs += [self.model_info['parameters']]

    # Add train data files
    inputs += [f"{self.train_data_input}/{i}_{self.train_data_split}.parquet" for i in ["X","wt"]]

    # Add validation data files
    inputs += [f"{self.nominal_data_input}/{i}_{self.sim_type}.parquet" for i in ["X","wt"]]
    inputs += [f"{self.up_data_input}/{i}_{self.sim_type}.parquet" for i in ["X","wt"]]
    inputs += [f"{self.down_data_input}/{i}_{self.sim_type}.parquet" for i in ["X","wt"]]

    return inputs

        