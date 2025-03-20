import copy

import numpy as np

from data_processor import DataProcessor
from plotting import plot_stacked_histogram_with_ratio, plot_stacked_unrolled_2d_histogram_with_ratio, plot_many_comparisons, plot_histograms_with_ratio
from useful_functions import Translate, LoadConfig

class Generator():

  def __init__(self):

    self.data_input = None
    self.asimov_input = None
    self.plots_output = "plots/"
    self.do_1d = True
    self.do_2d_unrolled = False
    self.do_transformed = False
    self.extra_plot_name = ""
    self.sim_type = "val"
    self.val_info = {}
    self.verbose = True

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"

  def Run(self):

    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg)

    sim_dps = {}
    synth_dps = {}

    val_ind_text = ", ".join([f"{Translate(k)}={round(v,2)}" for k, v in self.val_info.items()])

    for k in self.data_input.keys():

      print(f"- Calculating for {k}")

      # Make data processors
      if self.verbose:
        print(f"- Making data processor for {k}")

      sim_dps[k] = DataProcessor(
        [[f"{self.data_input[k]}/X_{self.sim_type}.parquet", f"{self.data_input[k]}/wt_{self.sim_type}.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {}
      )

      synth_dps[k] = DataProcessor(
        [[f"{self.asimov_input[k]}/asimov.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {}
      )

    # Get names for plot
    sim_plot_name = "Simulated"
    sample_plot_name = "Synthetic"


    # Running plotting functions
    if self.do_1d:
      if self.verbose:
        print(f"- Making 1D generation plots")

      self._PlotGeneration(
        synth_dps, 
        sim_dps, 
        cfg["variables"],
        sim_plot_name,
        sample_plot_name,
        transform=False,
        extra_dir="GenerationTrue1D",
        extra_name=self.extra_plot_name,
        axis_text=val_ind_text,
      )
      if self.do_transformed:
        self._PlotGeneration(
          synth_dps, 
          sim_dps, 
          cfg["variables"],
          sim_plot_name,
          sample_plot_name,
          transform=True,
          extra_dir="GenerationTrue1DTransformed",
          extra_name=self.extra_plot_name,
          axis_text=val_ind_text,
        )

    if self.do_2d_unrolled:
      if self.verbose:
        print(f"- Making 2D unrolled generation plots")

      self._Plot2DUnrolledGeneration(
        synth_dps, 
        sim_dps, 
        cfg["variables"],
        sim_plot_name,
        sample_plot_name,
        transform=False,
        extra_dir="GenerationTrue2DUnrolled",
        extra_name=self.extra_plot_name,
        axis_text=val_ind_text,      
      )

      if self.do_transformed:
        self._Plot2DUnrolledGeneration(
          synth_dps, 
          sim_dps, 
          cfg["variables"],
          sim_plot_name,
          sample_plot_name,
          transform=True,
          extra_dir="GenerationTrue2DUnrolledTransformed",
          extra_name=self.extra_plot_name,
          axis_text=val_ind_text,
        )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Add config
    outputs += [self.cfg]

    # Load config
    cfg = LoadConfig(self.cfg)

    # Loop through columns
    for col in cfg["variables"]:
      outputs += [f"{self.plots_output}/GenerationTrue1D/generation_{col}{self.extra_plot_name}_plot_style_{i}.pdf" for i in [1,2,3]]
      if self.do_transformed:
        outputs += [f"{self.plots_output}/GenerationTrue1DTransformed/generation_{col}{self.extra_plot_name}_plot_style_{i}.pdf" for i in [1,2,3]]
      if self.do_2d_unrolled:
        for plot_col in cfg["variables"]:
          if col == plot_col: continue
          outputs += [f"{self.plots_output}/GenerationTrue2DUnrolled/generation_unrolled_2d_{plot_col}_{col}{self.extra_plot_name}.pdf"]
          if self.do_transformed:
            outputs += [f"{self.plots_output}/GenerationTrue2DUnrolledTransformed/generation_unrolled_2d_{plot_col}_{col}{self.extra_plot_name}.pdf"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initiate inputs
    inputs = []

    # Add data files
    for k in self.data_input.keys():
      inputs += [
        f"{self.data_input[k]}/X_{self.sim_type}.parquet",
        f"{self.data_input[k]}/wt_{self.sim_type}.parquet",
        f"{self.asimov_input[k]}/asimov.parquet",
      ]

    return inputs

  def _PlotGeneration(
    self, 
    synth_dps, 
    sim_dps, 
    X_columns,
    sim_plot_name,
    synth_plot_name,
    n_bins=40, 
    transform=False,
    density=False,
    extra_dir="",
    extra_name="",
    axis_text="",
    ):


    if extra_dir != "":
      extra_dir = f"{extra_dir}/"

    functions_to_apply = []
    if transform: functions_to_apply.append("transform")

    for col in X_columns:

      # Get binning
      bins = sim_dps[list(sim_dps.keys())[0]].GetFull(
        method = "bins_with_equal_spacing", 
        functions_to_apply = functions_to_apply,
        bins = n_bins,
        column = col,
      )      

      # Make synth hists
      synth_hists = {}
      sim_hists = {}
      synth_hist_uncerts = {}
      sim_hist_uncerts = {}
      for ind, file_name in enumerate(synth_dps.keys()):

        synth_hist, synth_hist_uncert, bins = synth_dps[file_name].GetFull(
          method = "histogram_and_uncert",
          functions_to_apply = functions_to_apply,
          bins = bins,
          column = col,
        )
        synth_hists[f"{file_name} {synth_plot_name}"] = synth_hist
        synth_hist_uncerts[f"{file_name} {synth_plot_name}"] = synth_hist_uncert

        if ind == 0:
          synth_hist_uncert_squared = synth_hist_uncert**2
          synth_hist_total = copy.deepcopy(synth_hist)
        else:
          synth_hist_uncert_squared += (synth_hist_uncert**2)
          synth_hist_total += synth_hist

      # Make sim hists
      for ind, file_name in enumerate(sim_dps.keys()):
        sim_hist, sim_hist_uncert, bins = sim_dps[file_name].GetFull(
          method = "histogram_and_uncert",
          functions_to_apply = functions_to_apply,
          bins = bins,
          column = col,
        )
        sim_hists[f"{file_name} {sim_plot_name}"] = sim_hist
        sim_hist_uncerts[f"{file_name} {sim_plot_name}"] = sim_hist_uncert

        if ind == 0:
          sim_hist_uncert_squared = sim_hist_uncert**2
          sim_hist_total = copy.deepcopy(sim_hist)
        else:
          sim_hist_uncert_squared += (sim_hist_uncert**2)
          sim_hist_total += sim_hist

      if density:
        sum_sim_hist_total = np.sum(sim_hist_total)
        sum_synth_hist_total = np.sum(synth_hist_total)
        sim_hist_total /= sum_sim_hist_total
        sim_hists = {k : v/sum_sim_hist_total for k,v in sim_hists.items()}
        sim_hist_uncert = np.sqrt(sim_hist_uncert_squared)/sum_sim_hist_total
        sim_hist_uncerts = {k : v/sum_sim_hist_total for k,v in sim_hist_uncerts.items()}
        synth_hist_uncert = np.sqrt(synth_hist_uncert_squared)/sum_synth_hist_total
        synth_hists = {k : v/sum_synth_hist_total for k,v in synth_hists.items()}
        synth_hist_uncerts = {k : v/sum_synth_hist_total for k,v in synth_hist_uncerts.items()}
      else:
        sim_hist_uncert = np.sqrt(sim_hist_uncert_squared)
        synth_hist_uncert = np.sqrt(synth_hist_uncert_squared)       

      plot_stacked_histogram_with_ratio(
        sim_hist_total, 
        synth_hists, 
        bins, 
        data_name = sim_plot_name, 
        xlabel=Translate(col),
        ylabel="Events" if not density else "Density",
        name=f"{self.plots_output}/{extra_dir}generation_{col}{extra_name}_plot_style_1", 
        data_errors=sim_hist_uncert, 
        stack_hist_errors=synth_hist_uncert, 
        use_stat_err=False,
        axis_text=axis_text,
        )

      plot_histograms_with_ratio(
        [[sim_hists[list(sim_hists.keys())[ind]], synth_hists[list(synth_hists.keys())[ind]]] for ind in range(len(sim_hists.keys()))],
        [[sim_hist_uncerts[list(sim_hists.keys())[ind]], synth_hist_uncerts[list(synth_hists.keys())[ind]]] for ind in range(len(sim_hists.keys()))],
        [[list(sim_hists.keys())[ind], list(synth_hists.keys())[ind]] for ind in range(len(sim_hists.keys()))],
        bins,
        xlabel = Translate(col),
        ylabel="Events" if not density else "Density",
        name=f"{self.plots_output}/{extra_dir}generation_{col}{extra_name}_plot_style_2",    
      )

      plot_many_comparisons(
        sim_hists,
        synth_hists,
        sim_hist_uncerts,
        synth_hist_uncerts,
        bins,
        xlabel = Translate(col),
        ylabel="Events" if not density else "Density",
        name=f"{self.plots_output}/{extra_dir}generation_{col}{extra_name}_plot_style_3",       
        )


      if len(sim_hists.keys()) > 1:

        sim_hists[f"Total {sim_plot_name}"] = sim_hist_total
        sim_hist_uncerts[f"Total {sim_plot_name}"] = sim_hist_uncert
        synth_hists[f"Total {synth_plot_name}"] = synth_hist_total
        synth_hist_uncerts[f"Total {synth_plot_name}"] = synth_hist_uncert

        plot_many_comparisons(
          sim_hists,
          synth_hists,
          sim_hist_uncerts,
          synth_hist_uncerts,
          bins,
          xlabel = Translate(col),
          ylabel="Events" if not density else "Density",
          name=f"{self.plots_output}/{extra_dir}generation_{col}{extra_name}_plot_style_4",       
          )

  def _Plot2DUnrolledGeneration(
    self, 
    synth_dps, 
    sim_dps, 
    X_columns,
    sim_plot_name,
    synth_plot_name,
    n_unrolled_bins = 5,
    n_bins = 10, 
    transform=False,
    density=False,
    extra_dir="",
    extra_name="",
    axis_text="",
    ):


    if extra_dir != "":
      extra_dir = f"{extra_dir}/"

    functions_to_apply = []
    if transform: functions_to_apply.append("transform")

    for plot_col_ind, plot_col in enumerate(X_columns):

      # Get bins for plot_col
      plot_col_bins = sim_dps[list(sim_dps.keys())[0]].GetFull(
        method = "bins_with_equal_spacing", 
        functions_to_apply = functions_to_apply,
        bins = n_bins,
        column = plot_col,
      )

      for unrolled_col_ind, unrolled_col in enumerate(X_columns):

        # Skip if the same column
        if plot_col == unrolled_col: continue

        # Get bins for plot_col
        unrolled_col_bins = sim_dps[list(sim_dps.keys())[0]].GetFull(
          method = "bins_with_equal_stats", 
          functions_to_apply = functions_to_apply,
          bins = n_unrolled_bins,
          column = unrolled_col,
        )      

        synth_hists = {}

        for ind, file_name in enumerate(synth_dps.keys()):

          # Make histograms
          synth_hist, synth_hist_uncert, bins = synth_dps[file_name].GetFull(
            method = "histogram_2d_and_uncert",
            functions_to_apply = functions_to_apply,
            bins = [unrolled_col_bins, plot_col_bins],
            column = [unrolled_col, plot_col],
            )

          sim_hist, sim_hist_uncert, bins = sim_dps[file_name].GetFull(
            method = "histogram_2d_and_uncert",
            functions_to_apply = functions_to_apply,
            bins = [unrolled_col_bins, plot_col_bins],
            column = [unrolled_col, plot_col],
            )

          if ind == 0:
            synth_hist_uncert_squared = synth_hist_uncert**2
            sim_hist_uncert_squared = sim_hist_uncert**2
            sim_hist_total = copy.deepcopy(sim_hist)
          else:
            synth_hist_uncert_squared += (synth_hist_uncert**2)
            sim_hist_uncert_squared += (sim_hist_uncert**2)
            sim_hist_total += sim_hist

          synth_hists[f"{file_name} {synth_plot_name}"] = synth_hist

        plot_stacked_unrolled_2d_histogram_with_ratio(
          sim_hist_total, 
          synth_hists, 
          bins[1],
          bins[0], 
          Translate(unrolled_col),
          data_name = sim_plot_name, 
          xlabel=Translate(plot_col),
          ylabel="Events" if not density else "Density",
          name=f"{self.plots_output}/{extra_dir}generation_unrolled_2d_{plot_col}_{unrolled_col}{extra_name}", 
          data_hists_errors=np.sqrt(sim_hist_uncert_squared), 
          stack_hists_errors=np.sqrt(synth_hist_uncert_squared), 
          use_stat_err=False,
          axis_text=axis_text,
        )