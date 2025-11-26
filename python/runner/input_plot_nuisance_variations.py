import copy
import yaml

import numpy as np

from data_processor import DataProcessor
from plotting import plot_histograms_with_ratio
from useful_functions import LoadConfig, Translate, GetValidationDefaultIndex

class InputPlotNuisanceVariations():

  def __init__(self):
    """
    A class to plot the validation distributions
    """
    # Required input which is the location of a file
    self.cfg = None
    self.parameters = None

    # Other
    self.file_name = None
    self.verbose = True
    self.data_input = "data/"
    self.plots_output = "plots/"
    self.sim_type = "val"
    self.nuisance = None

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
    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg) 

    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)   

    # Get validation index
    default_val_ind = GetValidationDefaultIndex(cfg, self.file_name)

    # Default data processor
    default_dp = DataProcessor(
      [[f"{self.data_input}/val_ind_{default_val_ind}/X_{self.sim_type}.parquet", f"{self.data_input}/val_ind_{default_val_ind}/wt_{self.sim_type}.parquet"]], 
      "parquet",
      options = {
        "wt_name" : "wt",
      }
    )

    # Up data processor
    up_dp = DataProcessor(
      [[f"{self.data_input}/{self.nuisance}_up/X_{self.sim_type}.parquet", f"{self.data_input}/{self.nuisance}_up/wt_{self.sim_type}.parquet"]], 
      "parquet",
      options = {
        "wt_name" : "wt",
      }
    )   

    # Down data processor
    down_dp = DataProcessor(
      [[f"{self.data_input}/{self.nuisance}_down/X_{self.sim_type}.parquet", f"{self.data_input}/{self.nuisance}_down/wt_{self.sim_type}.parquet"]], 
      "parquet",
      options = {
        "wt_name" : "wt",
      }
    )

    # Plot variations
    if self.verbose:
      print("- Plotting nuisance variations")
    self._PlotVariations(
      default_dp,
      up_dp,
      down_dp,
      parameters["density"]["X_columns"],
      n_bins=40,
      extra_name_for_plot = f"_{self.sim_type}"
    )

      
  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initialise inputs
    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add parameters
    inputs + [self.parameters]  

    # Add data
    cfg = LoadConfig(self.cfg)
    default_val_ind = GetValidationDefaultIndex(cfg, self.file_name)
    inputs += [f"{self.data_input}/val_ind_{default_val_ind}/X_{self.sim_type}.parquet"]
    inputs += [f"{self.data_input}/val_ind_{default_val_ind}/wt_{self.sim_type}.parquet"]
    inputs += [f"{self.data_input}/{self.nuisance}_up/X_{self.sim_type}.parquet"]
    inputs += [f"{self.data_input}/{self.nuisance}_up/wt_{self.sim_type}.parquet"]
    inputs += [f"{self.data_input}/{self.nuisance}_down/X_{self.sim_type}.parquet"]
    inputs += [f"{self.data_input}/{self.nuisance}_down/wt_{self.sim_type}.parquet"]

    return inputs


  def _PlotVariations(self, nominal_dp, up_dp, down_dp, X_columns, n_bins=40, extra_name_for_plot=""):

    for col in X_columns:

      bins = nominal_dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, column=col, ignore_quantile=0.01, ignore_discrete=False)

      nominal_hist, nominal_hist_uncert, _ = nominal_dp.GetFull(method="histogram_and_uncert", bins=bins, column=col)
      up_hist, up_hist_uncert, _ = up_dp.GetFull(method="histogram_and_uncert", bins=bins, column=col)
      down_hist, down_hist_uncert, _ = down_dp.GetFull(method="histogram_and_uncert", bins=bins, column=col)

      plot_name = f"{self.plots_output}/nuisance_variation_{self.nuisance}_{col}{extra_name_for_plot}"

      plot_hists = [[up_hist,nominal_hist], [down_hist,nominal_hist]]
      plot_errs = [[np.zeros_like(up_hist_uncert), np.zeros_like(up_hist_uncert)], [np.zeros_like(down_hist_uncert), np.zeros_like(down_hist_uncert)]]
      plot_names = [["Up","Nominal"], ["Down","Nominal"]]

      plot_histograms_with_ratio(
        plot_hists,
        plot_errs,
        plot_names,
        bins,
        name = plot_name,
        xlabel = Translate(col),
        ylabel = "Events",
        anchor_y_at_0 = True,
        first_ratio = True,
        ratio_range = [0.9,1.1],
        axis_text = self.nuisance
      )