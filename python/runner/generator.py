import yaml
import copy
import numpy as np
import pandas as pd
from functools import partial

from useful_functions import GetYName
from plotting import plot_stacked_histogram_with_ratio, plot_stacked_unrolled_2d_histogram_with_ratio

class Generator():

  def __init__(self):

    self.parameters = None
    self.model = None
    self.architecture = None
    self.Y_sim = None
    self.Y_synth = None

    self.pois = None
    self.nuisances = None

    self.yield_function = "default"
    self.plots_output = "plots/"
    self.verbose = True
    self.n_synth = 10**6
    self.scale_to_yield = False
    self.do_1d = True
    self.do_2d_unrolled = False

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    # Make singular inputs as dictionaries
    combined_model = True
    if isinstance(self.model, str):
      combined_model = False
      with open(self.parameters, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
      self.model = {parameters['file_name'] : self.model}
      self.parameters = {parameters['file_name'] : self.parameters}
      self.architecture = {parameters['file_name'] : self.architecture}

  def Run(self):

    from data_processor import DataProcessor
    from network import Network
    from yields import Yields

    # Loop through and make networks
    networks = {}
    sim_dps = {}
    synth_dps = {}
    for file_name in self.model.keys():

      # Open parameters
      if self.verbose:
        print(f"- Loading in the parameters for model {file_name}")
      with open(self.parameters[file_name], 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Load the architecture in
      if self.verbose:
        print(f"- Loading in the architecture for model {file_name}")
      with open(self.architecture[file_name], 'r') as yaml_file:
        architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Build model
      if self.verbose:
        print(f"- Building the model for {file_name}")
      networks[parameters['file_name']] = Network(
        f"{parameters['file_loc']}/X_train.parquet",
        f"{parameters['file_loc']}/Y_train.parquet", 
        f"{parameters['file_loc']}/wt_train.parquet", 
        f"{parameters['file_loc']}/X_test.parquet",
        f"{parameters['file_loc']}/Y_test.parquet", 
        f"{parameters['file_loc']}/wt_test.parquet",
        options = {
          **architecture,
          **{
            "data_parameters" : parameters
          }
        }
      )  
      
      # Loading model
      if self.verbose:
        print(f"- Loading the model for {file_name}")
      networks[file_name].Load(name=self.model[file_name])

      # Make yields function
      if self.scale_to_yield:
        yields = Yields(
          pd.read_parquet(parameters['yield_loc']), 
          self.pois, 
          self.nuisances, 
          file_name,
          method=self.yield_function, 
          column_name="yield"
        )

      # Make data processors
      shape_Y_cols = [col for col in self.Y_sim.columns if "mu_" not in col and col in parameters["Y_columns"]]
      if self.verbose:
        print(f"- Making data processor for {file_name}")
      sim_dps[file_name] = DataProcessor(
        [[f"{parameters['file_loc']}/X_val.parquet", f"{parameters['file_loc']}/Y_val.parquet", f"{parameters['file_loc']}/wt_val.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : parameters,
          "selection" : " & ".join([f"({col}=={self.Y_sim.loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
          "scale" : yields.GetYield(self.Y_sim) if self.scale_to_yield else 1.0,
        }
      )

      synth_dps[file_name] = DataProcessor(
        [[partial(networks[file_name].Sample, self.Y_synth)]],
        "generator",
        n_events = self.n_synth,
        options = {
          "parameters" : parameters,
          "scale" : yields.GetYield(self.Y_synth) if self.scale_to_yield else 1.0,
        }
      )

    # Get names for plot
    sim_file_name = GetYName(self.Y_sim, purpose="file", prefix="_sim_y_")
    sample_file_name = GetYName(self.Y_synth, purpose="file", prefix="_synth_y_")
    sim_plot_name = GetYName(self.Y_sim, purpose="plot", prefix="Simulated y=")
    sample_plot_name = GetYName(self.Y_synth, purpose="plot", prefix="Synthetic y=")

    # Running plotting functions
    if self.do_1d:
      if self.verbose:
        print(f"- Making 1D generation plots")

      self._PlotGeneration(
        synth_dps, 
        sim_dps, 
        parameters["X_columns"],
        sim_file_name,
        sample_file_name,
        sim_plot_name,
        sample_plot_name,
        transform=False,
        extra_dir="GenerationTrue1D"
      )
      self._PlotGeneration(
        synth_dps, 
        sim_dps, 
        parameters["X_columns"],
        sim_file_name,
        sample_file_name,
        sim_plot_name,
        sample_plot_name,
        transform=True,
        extra_dir="GenerationTrue1DTransformed"
      )

    if self.do_2d_unrolled:
      if self.verbose:
        print(f"- Making 2D unrolled generation plots")

      self._Plot2DUnrolledGeneration(
        synth_dps, 
        sim_dps, 
        parameters["X_columns"],
        sim_file_name,
        sample_file_name,
        sim_plot_name,
        sample_plot_name,
        transform=False,
        extra_dir="GenerationTrue2DUnrolled"
      )

      self._Plot2DUnrolledGeneration(
        synth_dps, 
        sim_dps, 
        parameters["X_columns"],
        sim_file_name,
        sample_file_name,
        sim_plot_name,
        sample_plot_name,
        transform=True,
        extra_dir="GenerationTrue2DUnrolledTransformed"
      )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    file_name = list(self.models.keys())[0]
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    sim_file_name = GetYName(self.Y_sim, purpose="file", prefix="_sim_y_")
    synth_file_name = GetYName(self.Y_synth, purpose="file", prefix="_synth_y_")
    if self.do_1d:
      for col in parameters["X_columns"]:
        outputs += [
          f"{self.plots_output}/GenerationTrue1D/generation_{col}{sim_file_name}{synth_file_name}.pdf",
          f"{self.plots_output}/GenerationTrue1DTransform/generation_{col}{sim_file_name}{synth_file_name}.pdf",
        ]
    if self.do_2d_unrolled:
      for plot_col in parameters["X_columns"]:
        for unrolled_col in parameters["X_columns"]:
          if plot_col == unrolled_col: continue
          outputs += [
            f"{self.plots_output}/GenerationTrue2DUnrolled/generation_unrolled_2d_{plot_col}_{unrolled_col}{sim_file_name}{synth_file_name}.pdf", 
            f"{self.plots_output}/GenerationTrue2DUnrolledTransformed/generation_unrolled_2d_{plot_col}_{unrolled_col}{sim_file_name}{synth_file_name}.pdf", 
          ]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for file_name in self.model.keys():
      with open(self.parameters, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
      inputs += [
        self.model[file_name],
        self.architecture[file_name],
        self.parameters[file_name],
        f"{parameters['file_loc']}/X_train.parquet",
        f"{parameters['file_loc']}/Y_train.parquet", 
        f"{parameters['file_loc']}/wt_train.parquet", 
        f"{parameters['file_loc']}/X_test.parquet",
        f"{parameters['file_loc']}/Y_test.parquet", 
        f"{parameters['file_loc']}/wt_test.parquet",        
      ]

  def _PlotGeneration(
    self, 
    synth_dps, 
    sim_dps, 
    X_columns,
    sim_file_name,
    synth_file_name,
    sim_plot_name,
    synth_plot_name,
    n_bins=40, 
    transform=False,
    extra_dir="",
    extra_name=""
    ):

    if extra_dir != "":
      extra_dir = f"{extra_dir}/"

    functions_to_apply = []
    if not transform:
      functions_to_apply.append("untransform")
    else:
      functions_to_apply += ["untransform","transform"]

    for col in X_columns:

      synth_hists = {}

      for ind, file_name in enumerate(synth_dps.keys()):

        # Get binning
        if ind == 0:
          bins = sim_dps[file_name].GetFull(
            method = "bins_with_equal_spacing", 
            functions_to_apply = functions_to_apply,
            bins = n_bins,
            column = col,
          )

        # Make histograms
        synth_hist, synth_hist_uncert, bins = synth_dps[file_name].GetFull(
          method = "histogram_and_uncert",
          functions_to_apply = functions_to_apply,
          bins = bins,
          column = col,
          )

        sim_hist, sim_hist_uncert, bins = sim_dps[file_name].GetFull(
          method = "histogram_and_uncert",
          functions_to_apply = functions_to_apply,
          bins = bins,
          column = col,
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

      plot_stacked_histogram_with_ratio(
        sim_hist_total, 
        synth_hists, 
        bins, 
        data_name = sim_plot_name, 
        xlabel=col,
        ylabel="Events" if self.scale_to_yield else "Density",
        name=f"{self.plots_output}/{extra_dir}generation_{col}{sim_file_name}{synth_file_name}{extra_name}", 
        data_errors=np.sqrt(sim_hist_uncert_squared), 
        stack_hist_errors=np.sqrt(synth_hist_uncert_squared), 
        title_right="",
        use_stat_err=False,
        axis_text="",
        )

  def _Plot2DUnrolledGeneration(
    self, 
    synth_dps, 
    sim_dps, 
    X_columns,
    sim_file_name,
    synth_file_name,
    sim_plot_name,
    synth_plot_name,
    n_unrolled_bins = 5,
    n_bins = 10, 
    transform=False,
    extra_dir="",
    extra_name=""
    ):

    if extra_dir != "":
      extra_dir = f"{extra_dir}/"

    functions_to_apply = []
    if not transform:
      functions_to_apply.append("untransform")
    else:
      functions_to_apply += ["untransform","transform"]

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
          unrolled_col,
          data_name = sim_plot_name, 
          xlabel=plot_col,
          ylabel="Events" if self.scale_to_yield else "Density",
          name=f"{self.plots_output}/{extra_dir}generation_unrolled_2d_{plot_col}_{unrolled_col}{sim_file_name}{synth_file_name}{extra_name}", 
          data_hists_errors=np.sqrt(sim_hist_uncert_squared), 
          stack_hists_errors=np.sqrt(synth_hist_uncert_squared), 
          title_right="",
          use_stat_err=False,
          axis_text="",
        )


  def _Plot2DPulls(self, synth_dp, sim_dp):
    print()

  def _PlotCorrelationMatrix(self, synth_dp, sim_dp):
    print()