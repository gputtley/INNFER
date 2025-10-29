import copy
import yaml

import numpy as np

from data_processor import DataProcessor
from plotting import plot_histograms, plot_histograms_with_ratio
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
    self.sim_type = "val"
    self.category = None
    self.use_scenario_labels = False

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

    # Get defaults
    defaults = GetDefaultsInModel(parameters['file_name'], cfg, category=self.category)
    ratio_index = None
    for ind, val_dict in enumerate(self.val_loop):
      default = True
      for k, v in val_dict.items():
        if k in defaults.keys():
          if v != defaults[k]:
            default = False
      if default:
        ratio_index = copy.deepcopy(ind)
        break

    # Plots varying all validation parameters
    self._PlotVariations(self.val_loop, range(len(self.val_loop)), cfg["variables"], "X_distributions", n_bins=40, data_splits=[self.sim_type], ratio_index=ratio_index, columns=list(defaults.keys()))

    # plots varying one at a time and freezing others to the nominal
    for vary_name in defaults.keys():
      vary_ratio_index = None
      vary_val_loop = []
      vary_inds = []
      for ind, val_dict in enumerate(self.val_loop):
        include = True
        for k, v in val_dict.items():
          if k == vary_name: continue
          if k not in defaults.keys():
            include = False
            continue
          if v != defaults[k]:
            include = False
        if include:
          if vary_ratio_index is None and ratio_index is not None:
            if ind == ratio_index:
              vary_ratio_index = len(vary_val_loop)
          vary_val_loop.append(val_dict)
          vary_inds.append(ind)      

      if len(vary_val_loop) > 1:
        self._PlotVariations(vary_val_loop, vary_inds, cfg["variables"], f"X_distributions_varying_{vary_name}", n_bins=40, data_splits=[self.sim_type], ratio_index=vary_ratio_index, columns=list(defaults.keys()))

      
  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add plots
    defaults = GetDefaultsInModel(self.file_name, cfg, category=self.category)
    for col in cfg["variables"]:
      outputs += [f"{self.plots_output}/X_distributions_{col}.pdf"]
      for vary_name in defaults.keys():        
        vary_inds = []
        for ind, val_dict in enumerate(self.val_loop):
          include = True
          for k, v in val_dict.items():
            if k == vary_name: continue
            if k not in defaults.keys():
              include = False
              continue
            if v != defaults[k]:
              include = False
          if include:
            vary_inds.append(ind)      
        if len(vary_inds) > 1:
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
      inputs += [f"{self.data_input}/val_ind_{ind}/X_{self.sim_type}.parquet"]
      inputs += [f"{self.data_input}/val_ind_{ind}/Y_{self.sim_type}.parquet"]
      inputs += [f"{self.data_input}/val_ind_{ind}/wt_{self.sim_type}.parquet"]

    return inputs


  def _PlotVariations(self, variations, inds, X_columns, vary_name, n_bins=40, data_splits=["val"], extra_name_for_plot="", ratio_index=None, columns=None, scenario_inds=[]):

    for data_split in data_splits:

      hists = {}
      hist_names = {}
      hist_errs = {}
      bins = {}

      for ind, variation in enumerate(variations):

        dp = DataProcessor(
          [[f"{self.data_input}/val_ind_{inds[ind]}/X_{data_split}.parquet", f"{self.data_input}/val_ind_{inds[ind]}/Y_{data_split}.parquet", f"{self.data_input}/val_ind_{inds[ind]}/wt_{data_split}.parquet"]], 
          "parquet",
          options = {
            "wt_name" : "wt",
            "selection" : " & ".join([f"({k}=={v})" for k, v in variation.items() if columns is None or k in columns]) if len(variation.keys()) > 0 else None
          }
        )

        count = dp.GetFull(method="count")
        if count == 0: continue

        for col in X_columns:

          if col not in bins.keys():
            bins[col] = dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, column=col, ignore_quantile=0.01, ignore_discrete=False)

          hist, hist_uncert, _ = dp.GetFull(method="histogram_and_uncert", bins=bins[col], column=col)
          if col not in hists.keys():
            hists[col] = []
            hist_names[col] = []
            hist_errs[col] = []

          hists[col].append(hist)
          hist_errs[col].append(hist_uncert)
          if not self.use_scenario_labels:
            hist_names[col].append(", ".join([f"{k}={v}" for k, v in variation.items()]))
          else:
            hist_names[col].append(f"Scenario {inds[ind]+1}")


      for col in X_columns:

        plot_name = self.plots_output+f"/{vary_name}_{col}{extra_name_for_plot}"

        # Make distribution plot
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
          hist_errs = [hist_err/np.sum(hist) for hist_err in hist_errs[col]],
        )

        # Make ratio plots

        ratio_hist = hists[col][ratio_index] if ratio_index is not None else hists[col][0]
        ratio_hist_err = hist_errs[col][ratio_index] if ratio_index is not None else hist_errs[col][0]
        sum_ratio_hist = np.sum(ratio_hist)
        ratio_hist = ratio_hist/sum_ratio_hist
        ratio_hist_err = ratio_hist_err/sum_ratio_hist
        other_hists = [hists[col][ind] for ind in range(len(hists[col])) if ind != ratio_index] if ratio_index is not None else [hists[col][ind] for ind in range(1, len(hists[col]))]
        other_hists_err = [hist_errs[col][ind] for ind in range(len(hists[col])) if ind != ratio_index] if ratio_index is not None else [hist_errs[col][ind] for ind in range(1, len(hists[col]))]
        other_hists_err = [hist_err/np.sum(other_hists[ind]) for ind, hist_err in enumerate(other_hists_err)]
        other_hists = [hist/np.sum(hist) for hist in other_hists]

        if len(other_hists) == 0:
          continue

        plot_hists = [[other_hists[ind], ratio_hist] for ind in range(len(other_hists))]
        plot_errs = [[other_hists_err[ind], ratio_hist_err] for ind in range(len(other_hists_err))]
        plot_names = []
        for ind in range(len(hist_names[col])):
          if ratio_index is not None:
            if ind == ratio_index: continue
            plot_names.append([hist_names[col][ind], hist_names[col][ratio_index]])
          else:
            if ind == 0: continue
            plot_names.append([hist_names[col][ind], hist_names[col][0]])

        plot_histograms_with_ratio(
          plot_hists,
          plot_errs,
          plot_names,
          bins[col],
          name = plot_name+"_ratio",
          xlabel = Translate(col),
          ylabel = "Density",
          anchor_y_at_0 = True,
          first_ratio = True,
          ratio_range = [0.9,1.1]
        )


