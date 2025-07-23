import yaml

import numpy as np

from data_processor import DataProcessor
from plotting import plot_histograms
from useful_functions import LoadConfig, GetDefaultsInModel, Translate

class InputPlotValidation():

  def __init__(self):
    """
    A class to plot the validation distributions
    """
    # Required input which is the location of a file
    self.cfg = None
    self.parameters = None

    # Other
    self.file_name = None
    self.val_loop = []
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
    # Load config
    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg) 

    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)   

    # Plots varying all validation parameters
    self._PlotVariations(self.val_loop, range(len(self.val_loop)), cfg["variables"], "X_distributions", n_bins=40, data_splits=["val"])

    # plots varying one at a time and freezing others to the nominal
    defaults = GetDefaultsInModel(parameters['file_name'], cfg)
    for vary_name in defaults.keys():
      vary_val_loop = []
      vary_inds = []
      for ind, val_dict in enumerate(self.val_loop):
        include = True
        for k, v in val_dict.items():
          if k == vary_name: continue
          if v != defaults[k]:
            include = False
        if include:
          vary_val_loop.append(val_dict)
          vary_inds.append(ind)      

      self._PlotVariations(vary_val_loop, vary_inds, cfg["variables"], f"X_distributions_varying_{vary_name}", n_bins=40, data_splits=["val"])

      
  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add plots
    defaults = GetDefaultsInModel(self.file_name, cfg)
    for col in cfg["variables"]:
      outputs += [f"{self.plots_output}/X_distributions_{col}.pdf"]
      for vary_name in defaults.keys():        
        outputs += [f"{self.plots_output}/X_distributions_varying_{vary_name}_{col}.pdf"]

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
    for ind, _ in enumerate(self.val_loop):
      inputs += [f"{self.data_input}/val_ind_{ind}/X_val.parquet"]
      inputs += [f"{self.data_input}/val_ind_{ind}/Y_val.parquet"]
      inputs += [f"{self.data_input}/val_ind_{ind}/wt_val.parquet"]

    return inputs


  def _PlotVariations(self, variations, inds, X_columns, vary_name, n_bins=40, data_splits=["val"], extra_name_for_plot=""):

    for data_split in data_splits:

      hists = {}
      hist_names = {}
      bins = {}

      for ind, variation in enumerate(variations):

        dp = DataProcessor(
          [[f"{self.data_input}/val_ind_{inds[ind]}/X_{data_split}.parquet", f"{self.data_input}/val_ind_{inds[ind]}/Y_{data_split}.parquet", f"{self.data_input}/val_ind_{inds[ind]}/wt_{data_split}.parquet"]], 
          "parquet",
          options = {
            "wt_name" : "wt",
            "selection" : " & ".join([f"({k}=={v})" for k, v in variation.items()]) if len(variation.keys()) > 0 else None
          }
        )

        count = dp.GetFull(method="count")
        if count == 0: continue

        for col in X_columns:

          if col not in bins.keys():
            bins[col] = dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, column=col, ignore_quantile=0.01, ignore_discrete=False)

          hist, _ = dp.GetFull(method="histogram", bins=bins[col], column=col)
          if col not in hists.keys():
            hists[col] = []
            hist_names[col] = []

          hists[col].append(hist)
          hist_names[col].append(", ".join([f"{Translate(k)}={v}" for k, v in variation.items()]))


      for col in X_columns:

        plot_name = self.plots_output+f"/{vary_name}_{col}{extra_name_for_plot}"
        plot_histograms(
          bins[col][:-1],
          [hist/np.sum(hist) for hist in hists[col]],
          hist_names[col],
          title_right = "",
          name = plot_name,
          x_label = Translate(col),
          y_label = "Density",
          anchor_y_at_0 = True,
          drawstyle = "steps-mid",
        )
