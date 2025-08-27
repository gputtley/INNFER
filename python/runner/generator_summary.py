import copy
import os
import yaml

import numpy as np
import pandas as pd
import seaborn as sns

from functools import partial

from plotting import plot_histograms
from useful_functions import Translate, GetDefaultsInModel, LoadConfig

class GeneratorSummary():

  def __init__(self):

    self.cfg = None
    self.val_loop = []
    self.data_input = []
    self.asimov_input = []
    self.sim_type = "test_inf"
    self.plots_output = "plots/"
    self.extra_plot_name = ""
    self.file_name = None
    self.n_bins = 20
    self.val_inds = None
    self.verbose = True        

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """

    for key, value in options.items():
      setattr(self, key, value)

    if self.val_inds is not None:
      self.val_inds = [float(i) for i in self.val_inds.split(",")]

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"

  def Run(self):

    from data_processor import DataProcessor

    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg)

    val_loop = []
    for val_ind, val_info in enumerate(self.val_loop):
      if self.val_inds is not None:
        if val_ind in self.val_inds: 
          val_loop.append(val_info)
      else:
        val_loop.append(val_info)


    if self.verbose:
      print("- Making dataprocessors")    
    sim_dps = {}
    synth_dps = {}
    for val_ind, val_info in enumerate(val_loop):

      # Make data processors
      sim_dps[val_ind] = DataProcessor(
        [[f"{self.data_input[val_ind][k]}/X_{self.sim_type}.parquet", f"{self.data_input[val_ind][k]}/wt_{self.sim_type}.parquet"] for k in self.data_input[val_ind].keys()],
        "parquet",
        wt_name = "wt",
        options = {}
      )

      synth_dps[val_ind] = DataProcessor(
        [[f"{self.asimov_input[val_ind][k]}/asimov.parquet"] for k in self.data_input[val_ind].keys()],
        "parquet",
        wt_name = "wt",
        options = {}
      )

    if self.verbose:
      print("- Finding default val_ind")   

    # Find default val_ind
    defaults_in_model = GetDefaultsInModel(self.file_name, cfg, include_rate=True, include_lnN=True)
    default_val_ind = None
    for val_ind, val_info in enumerate(val_loop):
      default_val = True
      for k, v in val_info.items():
        if v != defaults_in_model[k]:
          default_val = False
          break
      if default_val:
        default_val_ind = val_ind
        break
    if default_val_ind is None:
      default_val_ind = 0


    if self.verbose:
      print("- Making histograms") 

    # Loop through columns
    for col in cfg["variables"]:

      bins = synth_dps[default_val_ind].GetFull(
        #method = "bins_with_equal_spacing", 
        method = "bins_with_equal_stats",
        bins = self.n_bins,
        column = col,
        ignore_quantile = 0.001
      )

      nom_synth_hist, _ = synth_dps[default_val_ind].GetFull(
        method = "histogram",
        bins = bins,
        column = col,
      )

      synth_hists = {}
      sim_hists = {}
      sim_hist_uncerts = {}
      names = []
      for val_ind, val_info in enumerate(val_loop):

        synth_hists[val_ind], _ = synth_dps[val_ind].GetFull(
          method = "histogram",
          bins = bins,
          column = col,
        )
        sim_hists[val_ind], sim_hist_uncerts[val_ind], _ = sim_dps[val_ind].GetFull(
          method = "histogram_and_uncert",
          bins = bins,
          column = col,
        )

        names.append(", ".join([f"{Translate(k)}={v}" for k, v in val_info.items()]))


      # Plot ratio summary
      y_label = f"Ratio to Synthetic {names[default_val_ind]}"
      error_bar_names = [f"Simulated ({names[default_val_ind]})"] + [None]*(len(sim_hists.keys())-1)

      colours = ["black"] + list(sns.color_palette("Set2", len(list(sim_hists.keys()))-1))
      hists = [i/nom_synth_hist for i in synth_hists.values()]
      hists = [hists[default_val_ind]] + hists[:default_val_ind] + hists[default_val_ind+1:]
      hist_names = [f"Synthetic ({names[default_val_ind]})"] + names[:default_val_ind] + names[default_val_ind+1:]

      error_bar_hists = [i/nom_synth_hist for i in sim_hists.values()]
      error_bar_hist_uncerts = [i/nom_synth_hist for i in sim_hist_uncerts.values()]
      error_bar_hists = [error_bar_hists[default_val_ind]] + error_bar_hists[:default_val_ind] + error_bar_hists[default_val_ind+1:]
      error_bar_hist_uncerts = [error_bar_hist_uncerts[default_val_ind]] + error_bar_hist_uncerts[:default_val_ind] + error_bar_hist_uncerts[default_val_ind+1:]
      error_bar_names = [f"Simulated ({names[default_val_ind]})"] + [None]*(len(error_bar_hists)-1)


      plot_histograms(
        np.array(bins[:-1]),
        hists,
        hist_names,
        colors=colours,
        x_label=Translate(col),
        name=f"{self.plots_output}/generation_summary_{col}{self.extra_plot_name}", 
        y_label = y_label,
        error_bar_hists = error_bar_hists,
        error_bar_names = error_bar_names,
        error_bar_hist_errs = error_bar_hist_uncerts,
        legend_right = True,
      )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add the output files
    for col in cfg["variables"]:
      outputs += [f"{self.plots_output}/generation_summary_{col}{self.extra_plot_name}.pdf"]
    
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initiate inputs
    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add the input files
    for val_ind, val_info in enumerate(self.val_loop):
      for k in self.data_input[val_ind].keys():
        inputs += [f"{self.data_input[val_ind][k]}/X_{self.sim_type}.parquet"]
        inputs += [f"{self.data_input[val_ind][k]}/wt_{self.sim_type}.parquet"]
        inputs += [f"{self.asimov_input[val_ind][k]}/asimov.parquet"]

    return inputs