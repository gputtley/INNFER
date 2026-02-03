import copy

import numpy as np

from data_processor import DataProcessor
from plotting import plot_stacked_histogram_with_ratio, plot_stacked_unrolled_2d_histogram_with_ratio, plot_many_comparisons, plot_histograms_with_ratio
from useful_functions import Translate, LoadConfig, RoundUnrolledBins

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
    self.plot_styles = [1]
    self.no_text = True
    self.data_label = "Simulated"
    self.stack_label = "Synthetic"
    self.include_postfit_uncertainty = False
    self.uncertainty_input = None
    self.use_expected_data_uncertainty = False
    self.plot_var_and_bins = None


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
    synth_uncert_dps = None

    if self.no_text:
      val_ind_text = ""
    else:
      val_ind_text = ", ".join([f"{Translate(k)}={round(v,2)}" for k, v in self.val_info.items()])


    # Make sim data processors
    for k in self.data_input.keys():
      if self.verbose:
        print(f"- Making data processor for sim {k}")
      sim_dps[k] = DataProcessor(
        [self.data_input[k]],
        "parquet",
        wt_name = "wt",
        options = {
          "check_wt" : True,
        }
      )

    # Make synth data processors
    for k in self.asimov_input.keys():
      if self.verbose:
        print(f"- Making data processor for synth {k}")
      synth_dps[k] = DataProcessor(
        [self.asimov_input[k]],
        "parquet",
        wt_name = "wt",
        options = {
          "check_wt" : True,
        }
      )


    # Postfit uncertainty synth data processors
    if self.include_postfit_uncertainty:
      synth_uncert_dps = {}
      for k, v in self.uncertainty_input.items():
        synth_uncert_dps[k] = {}
        for nuisance, nuisance_values in v.items():
          synth_uncert_dps[k][nuisance] = {}
          for nuisance_value, file_name in nuisance_values.items():
            if self.verbose:
              print(f"- Making data processor for synth {k} postfit uncertainty {nuisance} {nuisance_value}")
            synth_uncert_dps[k][nuisance][nuisance_value] = DataProcessor(
              [file_name],
              "parquet",
              wt_name = "wt",
              options = {
                "check_wt" : True,
              }
            )

    # Get names for plot
    sim_plot_name = self.data_label
    sample_plot_name = self.stack_label
    if sample_plot_name != "":
      sample_plot_name = f" {sample_plot_name}"


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
        synth_uncert_dps=synth_uncert_dps,
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
          synth_uncert_dps=synth_uncert_dps,
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

    # Load config
    cfg = LoadConfig(self.cfg)

    # Loop through columns
    for col in cfg["variables"]:

      if self.plot_var_and_bins is not None:
        var_name = self.plot_var_and_bins.split("[")[0].split("(")[0]
        if var_name != col:
          continue

      outputs += [f"{self.plots_output}/GenerationTrue1D/generation_{col}{self.extra_plot_name}_plot_style_{i}.pdf" for i in self.plot_styles]
      if self.do_transformed:
        outputs += [f"{self.plots_output}/GenerationTrue1DTransformed/generation_{col}{self.extra_plot_name}_plot_style_{i}.pdf" for i in self.plot_styles]
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

    # Add config
    inputs += [self.cfg]

    # Add data files
    for k in self.data_input.keys():
      inputs += self.data_input[k]

    for k in self.asimov_input.keys():
      inputs += self.asimov_input[k]

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
    synth_uncert_dps=None,
    ):


    if extra_dir != "":
      extra_dir = f"{extra_dir}/"

    functions_to_apply = []
    if transform: functions_to_apply.append("transform")

    for col in X_columns:

      # Check if binning already specified
      if self.plot_var_and_bins is not None:
        if "[" in self.plot_var_and_bins:
          var_name, bins_str = self.plot_var_and_bins.split("[")
          bin_str = bins_str.split("]")[0]
          bins = np.array([float(i) for i in bins_str.split(",")])
        elif "(" in self.plot_var_and_bins:
          var_name, bins_str = self.plot_var_and_bins.split("(")
          bin_str = bins_str.split(")")[0]
          b = [float(i) for i in bin_str.split(",")]
          bins = np.arange(b[0], b[1], b[2])

        if var_name != col:
          continue

      else:

        # Get binning
        bins = sim_dps[list(sim_dps.keys())[0]].GetFull(
          method = "bins_with_equal_spacing", 
          functions_to_apply = functions_to_apply,
          bins = n_bins,
          column = col,
          ignore_quantile = 0.01,
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
        synth_hists[f"{file_name}{synth_plot_name}"] = synth_hist
        synth_hist_uncerts[f"{file_name}{synth_plot_name}"] = synth_hist_uncert

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

      # Change synth hist uncertainties to asymmetric uncertainties
      sim_hist_uncert = np.sqrt(sim_hist_uncert_squared)
      synth_hist_uncert = np.sqrt(synth_hist_uncert_squared)
      synth_hist_uncert_up = np.sqrt(synth_hist_uncert_squared)
      synth_hist_uncert_down = np.sqrt(synth_hist_uncert_squared)

      # Make synth uncertainty hists
      if synth_uncert_dps is not None:
        if self.plot_styles != [1]:
          print("Warning: Systematic uncertainties are only supported for plot style 1. Using plot style 1.")
        # Make histograms
        synth_extra_hist_uncerts = {}
        total_nuisances = []
        for ind, file_name in enumerate(synth_uncert_dps.keys()):
          synth_extra_hist_uncerts[file_name] = {}
          for nuisance, nuisance_values in synth_uncert_dps[file_name].items():
            if nuisance not in total_nuisances: total_nuisances.append(nuisance)
            synth_extra_hist_uncerts[file_name][nuisance] = {}
            for nuisance_value, dp in nuisance_values.items():
              synth_extra_hist_uncerts[file_name][nuisance][nuisance_value] = dp.GetFull(
                method = "histogram_and_uncert",
                functions_to_apply = functions_to_apply,
                bins = bins,
                column = col,
              )[0]
        # Get the totals
        synth_extra_hist_uncerts_total = {}
        for nuisance in total_nuisances:
          synth_extra_hist_uncerts_total[nuisance] = {}
          for nuisance_value in ["up", "down"]:
            for file_name in synth_extra_hist_uncerts.keys():
              if nuisance not in synth_extra_hist_uncerts[file_name].keys():
                use_hist = synth_hists[f"{file_name}{synth_plot_name}"] 
              else:
                use_hist = synth_extra_hist_uncerts[file_name][nuisance][nuisance_value]
              if nuisance_value not in synth_extra_hist_uncerts_total[nuisance].keys():
                synth_extra_hist_uncerts_total[nuisance][nuisance_value] = copy.deepcopy(use_hist)
              else:
                synth_extra_hist_uncerts_total[nuisance][nuisance_value] += use_hist
        # Get difference from the nominal
        for nuisance in total_nuisances:
          for nuisance_value in ["up", "down"]:
            synth_extra_hist_uncerts_total[nuisance][nuisance_value] -= synth_hist_total
        # Get the sum in quadrature
        for nuisance in total_nuisances:
          min_vals = np.minimum.reduce([synth_extra_hist_uncerts_total[nuisance]["up"], synth_extra_hist_uncerts_total[nuisance]["down"], np.zeros_like(synth_extra_hist_uncerts_total[nuisance]["up"])])
          max_vals = np.maximum.reduce([synth_extra_hist_uncerts_total[nuisance]["up"], synth_extra_hist_uncerts_total[nuisance]["down"], np.zeros_like(synth_extra_hist_uncerts_total[nuisance]["up"])])
          synth_hist_uncert_up = np.sqrt(synth_hist_uncert_squared + max_vals**2)
          synth_hist_uncert_down = np.sqrt(synth_hist_uncert_squared + min_vals**2)


      if density:
        sum_sim_hist_total = np.sum(sim_hist_total)
        sum_synth_hist_total = np.sum(synth_hist_total)
        sim_hist_total /= sum_sim_hist_total
        sim_hists = {k : v/sum_sim_hist_total for k,v in sim_hists.items()}
        sim_hist_uncert = sim_hist_uncert/sum_sim_hist_total
        sim_hist_uncerts = {k : v/sum_sim_hist_total for k,v in sim_hist_uncerts.items()}
        synth_hist_uncert = synth_hist_uncert/sum_synth_hist_total
        synth_hists = {k : v/sum_synth_hist_total for k,v in synth_hists.items()}
        synth_hist_uncerts = {k : v/sum_synth_hist_total for k,v in synth_hist_uncerts.items()}    


      if self.use_expected_data_uncertainty:
        sim_hist_uncert = np.sqrt(sim_hist_total)

      if 1 in self.plot_styles:
        plot_stacked_histogram_with_ratio(
          sim_hist_total, 
          {" ".join([Translate(kc) for kc in k.split(" ")]) : v for k,v in synth_hists.items()}, 
          bins, 
          data_name = sim_plot_name, 
          xlabel=Translate(col),
          ylabel="Events" if not density else "Density",
          name=f"{self.plots_output}/{extra_dir}generation_{col}{extra_name}_plot_style_1", 
          data_errors=sim_hist_uncert, 
          stack_hist_errors=None,
          stack_hist_errors_asym={
            "up": synth_hist_uncert_up,
            "down": synth_hist_uncert_down,
          },
          use_stat_err=False,
          axis_text=axis_text,
          )

      if 2 in self.plot_styles:

        plot_histograms_with_ratio(
          [[sim_hists[list(sim_hists.keys())[ind]], synth_hists[list(synth_hists.keys())[ind]]] for ind in range(len(sim_hists.keys()))],
          [[sim_hist_uncerts[list(sim_hists.keys())[ind]], synth_hist_uncerts[list(synth_hists.keys())[ind]]] for ind in range(len(sim_hists.keys()))],
          [[" ".join([Translate(i) for i in list(sim_hists.keys())[ind].split(" ")]), " ".join([Translate(i) for i in list(synth_hists.keys())[ind].split(" ")])] for ind in range(len(sim_hists.keys()))],
          bins,
          xlabel = Translate(col),
          ylabel="Events" if not density else "Density",
          name=f"{self.plots_output}/{extra_dir}generation_{col}{extra_name}_plot_style_2",    
        )

      if 3 in self.plot_styles:
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


      if 4 in self.plot_styles:
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

      if 5 in self.plot_styles:

        plot_histograms_with_ratio(
          [[sim_hists[list(sim_hists.keys())[ind]], synth_hists[list(synth_hists.keys())[ind]]] for ind in range(len(sim_hists.keys()))],
          [[sim_hist_uncerts[list(sim_hists.keys())[ind]], synth_hist_uncerts[list(synth_hists.keys())[ind]]] for ind in range(len(sim_hists.keys()))],
          [[" ".join([Translate(i) for i in list(sim_hists.keys())[ind].split(" ")]), " ".join([Translate(i) for i in list(synth_hists.keys())[ind].split(" ")])] for ind in range(len(sim_hists.keys()))],
          bins,
          xlabel = Translate(col),
          ylabel="Events" if not density else "Density",
          name=f"{self.plots_output}/{extra_dir}generation_{col}{extra_name}_plot_style_5",
          draw_error_bars = True,
          draw_error_bar_caps = False,
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
        unrolled_col_bins = RoundUnrolledBins(unrolled_col_bins)

        synth_hists = {}
        for ind, file_name in enumerate(synth_dps.keys()):

          # Make histograms
          synth_hist, synth_hist_uncert, bins = synth_dps[file_name].GetFull(
            method = "histogram_2d_and_uncert",
            functions_to_apply = functions_to_apply,
            bins = [unrolled_col_bins, plot_col_bins],
            column = [unrolled_col, plot_col],
            )

          if ind == 0:
            synth_hist_uncert_squared = synth_hist_uncert**2
          else:
            synth_hist_uncert_squared += (synth_hist_uncert**2)

          synth_hists[f"{Translate(file_name)}{synth_plot_name}"] = synth_hist


        for ind, file_name in enumerate(sim_dps.keys()):

          # Make histograms
          sim_hist, sim_hist_uncert, bins = sim_dps[file_name].GetFull(
            method = "histogram_2d_and_uncert",
            functions_to_apply = functions_to_apply,
            bins = [unrolled_col_bins, plot_col_bins],
            column = [unrolled_col, plot_col],
            )

          if ind == 0:
            sim_hist_uncert_squared = sim_hist_uncert**2
            sim_hist_total = copy.deepcopy(sim_hist)
          else:
            sim_hist_uncert_squared += (sim_hist_uncert**2)
            sim_hist_total += sim_hist


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