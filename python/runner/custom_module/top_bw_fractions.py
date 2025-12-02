import copy
import os
import pandas as pd
import pickle
import yaml

import numpy as np
import seaborn as sns

from functools import partial
from scipy.interpolate import UnivariateSpline

from data_processor import DataProcessor
from plotting import plot_histograms, plot_histograms_with_ratio
from useful_functions import MakeDirectories, LoadConfig, GetCategoryLoop
from breit_wigner_reweighting import bw_reweight

data_dir = str(os.getenv("PREP_DATA_DIR"))
plots_dir = str(os.getenv("PLOTS_DIR"))

class top_bw_fractions():

  def __init__(self):
    """
    A class to preprocess the datasets and produce the data
    parameters yaml file as well as the train, test and
    validation datasets.
    """
    # Required input which is the location of a file
    self.cfg = None
    self.options = {}
    self.batch_size = 10**7


  def _ApplyBWReweight(self, df, m=172.5, l=1.32):
    """
    Apply the BW reweighting to the dataframe.
    Args:
        df (pd.DataFrame): The input dataframe.
        m (float): The mass of the top quark.
        l (float): The width of the top quark. 
    """
    # Apply the BW reweighting
    df.loc[:,self.bw_mass_name] = m*np.ones(len(df))
    mask = df.loc[:,self.gen_mass] > 0
    df.loc[mask,"wt"] = bw_reweight(df.loc[mask,:], mass_to=self.bw_mass_name, mass_from=self.mass_name, gen_mass=self.gen_mass).loc[:,"wt"]
    if self.gen_mass_other is not None:
      mask = df.loc[:,self.gen_mass_other] > 0
      df.loc[mask,"wt"] = bw_reweight(df.loc[mask,:], mass_to=self.bw_mass_name, mass_from=self.mass_name, gen_mass=self.gen_mass_other).loc[:,"wt"]
    return df


  def _ApplyFractions(self, df, fractions={}):
    """
    Apply the fractions to the dataframe.
    Args:
        df (pd.DataFrame): The input dataframe.
        fractions (dict): The fractions to apply.
    """
    # Apply the fractions
    df.loc[:,"wt"] *= df.loc[:,self.mass_name].map(fractions)
    return df  


  def _CalculateOptimalFractions(self, base_file, wt_func, selection=None):
    """
    Calculate the optimal fractions for the given base file and weight function.
    Args:
        base_file (str): The base file to use.
        wt_func (str): The weight function to use.
    Returns:
        tuple: A tuple containing the normalised fractions and the splines.
    """

    print("- Calculating optimal fractions")

    # full dataprocessor
    dp = DataProcessor(
      [[base_file]],
      "parquet",
      batch_size=self.batch_size,
      options = {
        "wt_name" : "wt",
        "selection" : selection
      }
    )

    # calculate weight
    def apply_wt(df, wt_func):
      if self.bw_mass_name not in df.columns:
        df.loc[:,self.bw_mass_name] = 172.5*np.ones(len(df))
      df.loc[:,"wt"] = df.eval(wt_func)
      return df
    apply_wt_partial = partial(apply_wt, wt_func=wt_func)

    # get unique Y and loop through
    unique = dp.GetFull(method="unique")[self.mass_name]

    # get nominal sum of weights
    nominal_sum_wt = []
    for transformed_from in unique:
      nominal_sum_wt.append(dp.GetFull(
          method = "sum", 
          extra_sel = f"{self.mass_name}=={transformed_from}", 
          functions_to_apply = [
            apply_wt_partial,
          ]
        )
      )

    # get nominal sum of weights spline
    nominal_sum_wt_splines = UnivariateSpline(unique, nominal_sum_wt, s=0, k=1)

    # get sum of weights and sum of weights squared after BW
    sum_wt = {}
    sum_wt_squared = {}
    for transformed_to in self.transformed_to_masses:
      sum_wt[transformed_to] = {}
      sum_wt_squared[transformed_to] = {}
      for transformed_from in unique:

        sum_wt[transformed_to][transformed_from] = dp.GetFull(
          method = "sum", 
          extra_sel = f"{self.mass_name}=={transformed_from}", 
          functions_to_apply = [
            apply_wt_partial,
            partial(self._ApplyBWReweight,m=transformed_to)
          ]
        )
        sum_wt_squared[transformed_to][transformed_from] = dp.GetFull(
          method = "sum_w2", 
          extra_sel = f"{self.mass_name}=={transformed_from}", 
          functions_to_apply = [
            apply_wt_partial,
            partial(self._ApplyBWReweight,m=transformed_to)
          ]
        )

    # Effective events
    eff_events = {}
    for transformed_to in self.transformed_to_masses:
      eff_events[transformed_to] = {}
      for transformed_from in unique:
        if sum_wt_squared[transformed_to][transformed_from] > 0:
          eff_events[transformed_to][transformed_from] = (sum_wt[transformed_to][transformed_from]**2) / sum_wt_squared[transformed_to][transformed_from]
        else:
          eff_events[transformed_to][transformed_from] = 0.0

    # Total effective events for transformed_to
    total_eff_events = {}
    for transformed_to in self.transformed_to_masses:
      total_eff_events[transformed_to] = np.sum(np.array(list(eff_events[transformed_to].values())))

    # Fraction of effective events
    eff_events_fraction = {}
    for transformed_to in self.transformed_to_masses:
      eff_events_fraction[transformed_to] = {}
      for transformed_from in unique:
        if total_eff_events[transformed_to] > 0:
          eff_events_fraction[transformed_to][transformed_from] = eff_events[transformed_to][transformed_from] / total_eff_events[transformed_to]
        else:
          eff_events_fraction[transformed_to][transformed_from] = 0.0

    # derive fractions
    fractions = {}
    for transformed_to in self.transformed_to_masses:
      fractions[transformed_to] = {}
      for transformed_from in unique:
        if eff_events_fraction[transformed_to][transformed_from] > self.ignore_quantile:
          fractions[transformed_to][transformed_from] = (sum_wt[transformed_to][transformed_from] / sum_wt_squared[transformed_to][transformed_from])
        else:
          fractions[transformed_to][transformed_from] = 0.0

    # derive normalisation
    normalised_fractions = {}
    for transformed_to in self.transformed_to_masses:
      normalised_fractions[transformed_to] = {}
      total_sum = np.sum(np.array(list(sum_wt[transformed_to].values())) * np.array(list(fractions[transformed_to].values())))
      for transformed_from in unique:
        normalised_fractions[transformed_to][transformed_from] = fractions[transformed_to][transformed_from] * nominal_sum_wt_splines(transformed_to) / total_sum


    # fit splines for continuous fractioning
    splines = {}
    masses = list(normalised_fractions.keys())
    for transformed_from in unique:
      fractions_to = [normalised_fractions[transformed_to][transformed_from] for transformed_to in self.transformed_to_masses]      
      splines[transformed_from] = UnivariateSpline(masses, fractions_to, s=0, k=1)

    return normalised_fractions, splines


  def _PlotReweighting(self, normalised_fractions, base_file, wt_func, selection=None, extra_name=None):
    """
    Plot the reweighting of the samples.
    Args:
        normalised_fractions (dict): The normalised fractions to plot.
        base_file (str): The base file to use.
        wt_func (str): The weight function to use.
    """

    print("- Plotting optimally reweighted samples")

    # full dataprocessor
    dp = DataProcessor(
      [[base_file]],
      "parquet",
      batch_size=self.batch_size,
      options = {
        "wt_name" : "wt",
        "selection" : selection
      }
    )

    # calculate weight
    def apply_wt(df, wt_func):
      if self.bw_mass_name not in df.columns:
        df.loc[:,self.bw_mass_name] = 172.5*np.ones(len(df))
      df.loc[:,"wt"] = df.eval(wt_func)
      return df
    apply_wt_partial = partial(apply_wt, wt_func=wt_func)

    unique = dp.GetFull(method="unique")

    for col in self.plot_columns:
      
      hists = []
      hist_names = []
      hist_uncerts = []
      drawstyles = []
      colours = []
      error_bar_hists = []
      error_bar_hist_uncerts = []
      error_bar_hist_names = []

      bins = dp.GetFull(
        method="bins_with_equal_spacing", 
        bins=40,
        column=col,
        ignore_quantile=0.02,
        functions_to_apply = [apply_wt_partial]
      )

      colour_list = sns.color_palette("Set2", len(unique[self.mass_name]))

      for i, mass in enumerate(unique[self.mass_name]):

        var_hist, var_hist_uncert, _ = dp.GetFull(
          method="histogram_and_uncert", 
          extra_sel=f"{self.mass_name}=={mass}",
          bins=bins,
          column=col,
          functions_to_apply = [apply_wt_partial]
        )

        integral = np.sum(var_hist*(bins[1]-bins[0]))
        var_hist /= integral
        var_hist_uncert /= integral
        hists.append(var_hist)
        hist_uncerts.append(var_hist_uncert)
        if i == 0:
          hist_names.append(r"Nominal ($m_{t}$ = " + f"{mass} GeV)")
        else:
          hist_names.append(r"$m_{t}$ = " + f"{mass} GeV")
        drawstyles.append("steps-mid")
        colours.append(colour_list[i])

        bw_reweighted_hist, bw_reweighted_hist_uncert, _ = dp.GetFull(
          method="histogram_and_uncert", 
          bins=bins,
          column=col,
          functions_to_apply=[
            apply_wt_partial,
            partial(self._ApplyBWReweight,m=mass),partial(self._ApplyFractions, fractions=normalised_fractions[mass])
          ]
        )

        integral = np.sum(bw_reweighted_hist*(bins[1]-bins[0]))
        bw_reweighted_hist /= integral
        bw_reweighted_hist_uncert /= integral
        error_bar_hists.append(bw_reweighted_hist)
        error_bar_hist_uncerts.append(bw_reweighted_hist_uncert)
        if i == 0:
          error_bar_hist_names.append(r"BW ($m_{t}$ = " + f"{mass} GeV)")
        else:
          error_bar_hist_names.append(None)

      MakeDirectories(self.plot_dir)

      if extra_name is not None:
        plot_name = f"{self.plot_dir}/bw_reweighted_{col}_{extra_name}"
      else:
        plot_name = f"{self.plot_dir}/bw_reweighted_{col}"


      plot_histograms(
        np.array(bins[:-1]), 
        hists, 
        hist_names, 
        hist_errs = hist_uncerts,
        error_bar_hists = error_bar_hists,
        error_bar_hist_errs = error_bar_hist_uncerts,
        error_bar_names = error_bar_hist_names,
        drawstyle=drawstyles, 
        colors=colours, 
        name=plot_name, 
        x_label=col, 
        y_label="Density"
      )

      plot_histograms_with_ratio(
        [[error_bar_hists[ind], hists[ind]] for ind in range(len(hists))[::-1]],
        [[error_bar_hist_uncerts[ind], hist_uncerts[ind]] for ind in range(len(hists))[::-1]],
        [[error_bar_hist_names[ind], hist_names[ind]] for ind in range(len(hists))[::-1]],
        np.array(bins),
        xlabel = col,
        ylabel = "Density",
        name = plot_name + "_ratio",
        ratio_range = [0.9,1.1],
        draw_error_bars = True,
      )


  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    self.plot = False if "plot" not in self.options else self.options["plot"].strip() == "True"
    self.base_file_name = "base_ttbar_$CATEGORY" if "base_file_name" not in self.options else self.options["base_file_name"]
    self.transformed_to_masses = [166.5,167.5,168.5,169.5,170.5,171.5,172.5,173.5,174.5,175.5,176.5,177.5,178.5]
    self.gen_mass = "GenTop1_mass" if "gen_mass" not in self.options else self.options["gen_mass"].strip()
    self.gen_mass_other = "GenTop2_mass" if "gen_mass_other" not in self.options else self.options["gen_mass_other"].strip()
    self.file_name = "ttbar" if "file_name" not in self.options else self.options["file_name"].strip()
    self.mass_name = "sim_mass" if "mass_name" not in self.options else self.options["mass_name"].strip()
    self.bw_mass_name = "bw_mass" if "bw_mass_name" not in self.options else self.options["bw_mass_name"].strip()
    self.ignore_quantile = 0.1 if "ignore_quantile" not in self.options else float(self.options["ignore_quantile"])

    cfg = LoadConfig(self.cfg)
    self.plot_columns = [self.gen_mass] + cfg["variables"]
    if self.gen_mass_other is not None:
      self.plot_columns.append(self.gen_mass_other)
    self.plot_dir = f"{plots_dir}/{cfg['name']}/top_bw_fractions/{self.file_name}"


  def Run(self):
    """
    Run the class.
    """
    # Load the config
    cfg = LoadConfig(self.cfg)

    splines = {}

    for category in GetCategoryLoop(cfg):

      base_file_name = self.base_file_name.replace("$CATEGORY",category)
      base_file = f"{data_dir}/{cfg['name']}/LoadData/{base_file_name}.parquet"

      # Calculate optimal fractions
      normalised_fractions, splines[category] = self._CalculateOptimalFractions(base_file, cfg["files"][base_file_name]["weight"], selection=cfg["categories"][category])

      # Plot reweighting
      if self.plot:
        self._PlotReweighting(normalised_fractions, base_file, cfg["files"][base_file_name]["weight"], selection=cfg["categories"][category], extra_name=category)

    # Write splines to file
    file_names = {}
    for k1, v1 in splines.items():
      file_names[k1] = {}
      for k2 in v1.keys():
        file_name = f"{data_dir}/{cfg['name']}/top_bw_fractions/spline_{k1}_{str(k2).replace('.','p')}.pkl"
        MakeDirectories(os.path.dirname(file_name))
        with open(file_name, "wb") as f:
          pickle.dump(v1[k2], f)
        file_names[k1][k2] = file_name

    # Save to yaml
    output_yaml = f"{data_dir}/{cfg['name']}/top_bw_fractions/top_bw_fraction_locations.yaml"
    MakeDirectories(os.path.dirname(output_yaml))
    with open(output_yaml, "w") as f:
      yaml.dump(file_names, f)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add plots
    if self.plot:
      for cat in GetCategoryLoop(cfg):
        for col in self.plot_columns:
          outputs += [f"{self.plot_dir}/bw_reweighted_{col}_{cat}.pdf"]

    # Add yaml file
    outputs += [f"{data_dir}/{cfg['name']}/top_bw_fractions/top_bw_fraction_locations.yaml"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Add config
    inputs = [
      self.cfg,
      ]

    # Load config
    cfg = LoadConfig(self.cfg)

    for category in GetCategoryLoop(cfg):

      base_file_name = self.base_file_name.replace("$CATEGORY",category)
      base_file = f"{data_dir}/{cfg['name']}/LoadData/{base_file_name}.parquet"
      inputs += [base_file]

    return inputs