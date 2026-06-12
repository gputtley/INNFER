import copy
import yaml

import numpy as np
import pandas as pd

from data_processor import DataProcessor
from plotting import plot_histograms_with_ratio
from useful_functions import LoadConfig, Translate, GetValidationDefaultIndex, GetDefaultsInModel

class InputPlotNuisanceVariations():

  def __init__(self):
    """
    A class to plot the validation distributions
    """
    # Required input which is the location of a file
    self.cfg = None
    self.open_cfg = None
    self.parameters = None

    # Other
    self.file_name = None
    self.verbose = True
    self.data_input = "data/"
    self.plots_output = "plots/"
    self.sim_type = "val"
    self.nuisance = None
    self.ratio_range = [0.9,1.1]
    self.binned = False
    self.category = None

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
      cfg = LoadConfig(self.cfg)

    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)   

    # If binned load in the relevant stuff
    if self.binned:

      from infer import Infer
      infer_class = Infer()
      infer_class.Configure(
        {
          "parameters" : {self.category : {self.file_name: self.parameters}},
          "binned_fit_morph_col" : cfg["pois"][0],
          "likelihood_type" : "binned_extended",
          "inference_options" : {
            "rate_parameters" : []
          },
        }
      )
      binned_yields = infer_class._BuildBinYields()[self.category][self.file_name]

      nominal_Y = GetDefaultsInModel(self.file_name, cfg, include_lnN=True)
      nominal_hist = binned_yields(pd.DataFrame({k: [v] for k, v in nominal_Y.items()}))
      up_Y = copy.deepcopy(nominal_Y)
      down_Y = copy.deepcopy(nominal_Y)
      up_Y[self.nuisance] = 1.0
      down_Y[self.nuisance] = -1.0
      up_hist = binned_yields(pd.DataFrame({k: [v] for k, v in up_Y.items()}))
      down_hist = binned_yields(pd.DataFrame({k: [v] for k, v in down_Y.items()}))
      bins = cfg["inference"]["binned_fit"]["input"][self.category]["binning"]
      var = cfg["inference"]["binned_fit"]["input"][self.category]["variable"]

      # Plot variations
      if self.verbose:
        print("- Plotting nuisance variations")
      self._PlotVariations(
        None,
        None,
        None,
        [var],
        extra_name_for_plot = f"_binned_{self.sim_type}",
        bins = bins,
        nominal_hist = nominal_hist,
        nominal_hist_uncert = np.zeros_like(nominal_hist),
        up_hist = up_hist,
        up_hist_uncert = np.zeros_like(up_hist),
        down_hist = down_hist,
        down_hist_uncert = np.zeros_like(down_hist),
      )

    else:

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
        extra_name_for_plot = f"_{self.sim_type}",
      )

      
  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      cfg = LoadConfig(self.cfg)

    if not self.binned:
      for col in cfg["variables"]:
        outputs += [f"{self.plots_output}/nuisance_variation_{self.nuisance}_{col}_{self.sim_type}.pdf"]
    else:
      col = cfg["inference"]["binned_fit"]["input"][self.category]["variable"]
      outputs += [f"{self.plots_output}/nuisance_variation_{self.nuisance}_{col}_binned_{self.sim_type}.pdf"]

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
    inputs += [self.parameters]  

    # Add data
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      cfg = LoadConfig(self.cfg)

    if not self.binned:
      default_val_ind = GetValidationDefaultIndex(cfg, self.file_name)
      inputs += [f"{self.data_input}/val_ind_{default_val_ind}/X_{self.sim_type}.parquet"]
      inputs += [f"{self.data_input}/val_ind_{default_val_ind}/wt_{self.sim_type}.parquet"]
      inputs += [f"{self.data_input}/{self.nuisance}_up/X_{self.sim_type}.parquet"]
      inputs += [f"{self.data_input}/{self.nuisance}_up/wt_{self.sim_type}.parquet"]
      inputs += [f"{self.data_input}/{self.nuisance}_down/X_{self.sim_type}.parquet"]
      inputs += [f"{self.data_input}/{self.nuisance}_down/wt_{self.sim_type}.parquet"]
      
    return inputs


  def _PlotVariations(
    self, 
    nominal_dp, 
    up_dp, 
    down_dp, 
    X_columns, 
    n_bins=40, 
    extra_name_for_plot="",
    bins = None,
    nominal_hist=None,
    nominal_hist_uncert=None,
    up_hist=None,
    up_hist_uncert=None,
    down_hist=None,
    down_hist_uncert=None,
  ):

    for col in X_columns:

      if bins is None:
        bins = nominal_dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, column=col, ignore_quantile=0.01, ignore_discrete=False)

      if nominal_hist is None:
        nominal_hist, nominal_hist_uncert, _ = nominal_dp.GetFull(method="histogram_and_uncert", bins=bins, column=col)
      if up_hist is None:
        up_hist, up_hist_uncert, _ = up_dp.GetFull(method="histogram_and_uncert", bins=bins, column=col)
      if down_hist is None:
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
        ratio_range = self.ratio_range,
        axis_text = self.nuisance
      )

      # Normalise to make a density version of the plot
      nominal_hist_density = nominal_hist / np.sum(nominal_hist)
      up_hist_density = up_hist / np.sum(up_hist)
      down_hist_density = down_hist / np.sum(down_hist)
      plot_name = f"{self.plots_output}/nuisance_variation_{self.nuisance}_{col}_density{extra_name_for_plot}"
      plot_hists = [[up_hist_density,nominal_hist_density], [down_hist_density,nominal_hist_density]]
      plot_errs = [[np.zeros_like(up_hist_uncert), np.zeros_like(up_hist_uncert)], [np.zeros_like(down_hist_uncert), np.zeros_like(down_hist_uncert)]]
      plot_names = [["Up","Nominal"], ["Down","Nominal"]]
      plot_histograms_with_ratio(
        plot_hists,
        plot_errs,
        plot_names,
        bins,
        name = plot_name,
        xlabel = Translate(col),
        ylabel = "Density",
        anchor_y_at_0 = True,
        first_ratio = True,
        ratio_range = self.ratio_range,
        axis_text = self.nuisance
      )