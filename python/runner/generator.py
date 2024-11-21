import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml

import numpy as np
import pandas as pd

from functools import partial

from plotting import plot_stacked_histogram_with_ratio, plot_stacked_unrolled_2d_histogram_with_ratio, plot_many_comparisons, plot_histograms_with_ratio
from useful_functions import GetYName

class Generator():

  def __init__(self):

    self.parameters = None
    self.model = None
    self.architecture = None
    self.Y_sim = None
    self.Y_synth = None

    self.pois = None
    self.nuisances = None

    self.split_nuisance_models = False
    self.yield_function = "default"
    self.plots_output = "plots/"
    self.verbose = True
    self.n_synth = 10**6
    self.scale_to_yield = False
    self.do_1d = True
    self.do_2d_unrolled = False
    self.do_transformed = False
    self.seed = 42
    self.data_type = "sim"
    self.extra_plot_name = ""
    self.other_input_files = []
    self.other_output_files = []
    self.data_file = None

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

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"

  def Run(self):

    from data_processor import DataProcessor
    from network import Network
    from yields import Yields

    # Loop through and make networks
    networks = {}
    sim_dps = {}
    synth_dps = {}
    for file_name in self.model.keys():

      # Check to see if model in use
      if self.split_nuisance_models:
        drop = False

        # If key in nuisances, and nuisance value is 0, and this is a model for that nuisance
        for key in self.Y_synth.columns:
          if key in self.nuisances and float(self.Y_synth.loc[0,key]) == 0 and file_name.endswith(key):
            drop = True

        # If all the nuisances are not equal to 0, and this is a nominal model, and a model for that nuisance exists.
        nuisances_not_equal_to_0 = not (self.Y_synth.loc[0,[i for i in self.nuisances if i in self.Y_synth.columns]] == 0).all()
        nominal_model = not any(file_name.endswith(nui) for nui in self.nuisances)
        nuisance_exists = f"{file_name}_{self.Y_synth.loc[0,[i for i in self.nuisances if i in self.Y_synth.columns]].idxmax()}" in self.model.keys()
        if nuisances_not_equal_to_0 and nominal_model and nuisance_exists:
          drop = True
        if drop:
          continue

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
        f"{parameters['file_loc']}/X_val.parquet",
        f"{parameters['file_loc']}/Y_val.parquet", 
        f"{parameters['file_loc']}/wt_val.parquet", 
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
      yields = Yields(
        pd.read_parquet(parameters['yield_loc']), 
        self.pois, 
        self.nuisances, 
        file_name,
        method=self.yield_function, 
        column_name="yield"
      )

      # Make data processors
      shape_Y_cols = [col for col in self.Y_synth.columns if "mu_" not in col and col in parameters["Y_columns"]]
      if self.verbose:
        print(f"- Making data processor for {file_name}")

      if self.data_type == "sim":

        sim_dps[file_name] = DataProcessor(
          [[f"{parameters['file_loc']}/X_val.parquet", f"{parameters['file_loc']}/Y_val.parquet", f"{parameters['file_loc']}/wt_val.parquet"]],
#          [[f"{parameters['file_loc']}/X_train.parquet", f"{parameters['file_loc']}/Y_train.parquet", f"{parameters['file_loc']}/wt_train.parquet"]],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : parameters,
            "selection" : " & ".join([f"({col}=={self.Y_sim.loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
            "scale" : yields.GetYield(self.Y_sim),
            "functions" : ["untransform"]
          }
        )

      elif self.data_type == "asimov":

        sim_dps[file_name] = DataProcessor(
          [[partial(networks[file_name].Sample, self.Y_sim)]],
          "generator",
          n_events = self.n_synth,
          options = {
            "parameters" : parameters,
            "scale" : yields.GetYield(self.Y_sim),
          }
        )

      synth_dps[file_name] = DataProcessor(
        [[partial(networks[file_name].Sample, self.Y_synth)]],
        "generator",
        n_events = self.n_synth,
        options = {
          "parameters" : parameters,
          "scale" : yields.GetYield(self.Y_synth),
        }
      )

    if self.data_type == "data":
      sim_dps["Data"] = DataProcessor(
        [[self.data_file]],
        "parquet",
      )

    # Get names for plot
    if self.data_type == "sim":
      sim_plot_name = GetYName(self.Y_sim, purpose="plot", prefix="Simulated y=")
    elif self.data_type == "asimov":
      sim_plot_name = GetYName(self.Y_sim, purpose="plot", prefix="Asimov y=")
    else:
      sim_plot_name = "Data"
    sample_plot_name = GetYName(self.Y_synth, purpose="plot", prefix="Synthetic y=")

    # Running plotting functions
    if self.do_1d:
      if self.verbose:
        print(f"- Making 1D generation plots")

      self._PlotGeneration(
        synth_dps, 
        sim_dps, 
        parameters["X_columns"],
        sim_plot_name,
        sample_plot_name,
        transform=False,
        extra_dir="GenerationTrue1D",
        extra_name=self.extra_plot_name,
      )
      if self.do_transformed:
        if self.data_type != "data":
          self._PlotGeneration(
            synth_dps, 
            sim_dps, 
            parameters["X_columns"],
            sim_plot_name,
            sample_plot_name,
            transform=True,
            extra_dir="GenerationTrue1DTransformed",
            extra_name=self.extra_plot_name,
          )

    if self.do_2d_unrolled:
      if self.verbose:
        print(f"- Making 2D unrolled generation plots")

      self._Plot2DUnrolledGeneration(
        synth_dps, 
        sim_dps, 
        parameters["X_columns"],
        sim_plot_name,
        sample_plot_name,
        transform=False,
        extra_dir="GenerationTrue2DUnrolled",
        extra_name=self.extra_plot_name,
      )

      if self.do_transformed:
        if self.data_type != "data":
          self._Plot2DUnrolledGeneration(
            synth_dps, 
            sim_dps, 
            parameters["X_columns"],
            sim_plot_name,
            sample_plot_name,
            transform=True,
            extra_dir="GenerationTrue2DUnrolledTransformed",
            extra_name=self.extra_plot_name,
          )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    file_name = list(self.model.keys())[0]
    with open(self.parameters[file_name], 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if self.do_1d:
      for col in parameters["X_columns"]:
        outputs += [
          f"{self.plots_output}/GenerationTrue1D/generation_{col}{self.extra_plot_name}.pdf",
        ]
        if self.data_type != "data":
          if self.do_transformed:
            outputs += [
              f"{self.plots_output}/GenerationTrue1DTransformed/generation_{col}{self.extra_plot_name}.pdf",
            ]

    if self.do_2d_unrolled:
      for plot_col in parameters["X_columns"]:
        for unrolled_col in parameters["X_columns"]:
          if plot_col == unrolled_col: continue
          outputs += [
            f"{self.plots_output}/GenerationTrue2DUnrolled/generation_unrolled_2d_{plot_col}_{unrolled_col}{self.extra_plot_name}.pdf", 
          ]
          if self.data_type != "data":
            if self.do_transformed:
              outputs += [
                f"{self.plots_output}/GenerationTrue2DUnrolledTransformed/generation_unrolled_2d_{plot_col}_{unrolled_col}{self.extra_plot_name}.pdf", 
              ]

    # Add other outputs
    outputs += self.other_output_files

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for file_name in self.model.keys():
      with open(self.parameters[file_name], 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
      inputs += [
        self.model[file_name],
        self.architecture[file_name],
        self.parameters[file_name],
        f"{parameters['file_loc']}/X_val.parquet",
        f"{parameters['file_loc']}/Y_val.parquet", 
        f"{parameters['file_loc']}/wt_val.parquet", 
      ]

    # Add inputs from the dataset being used
    if self.data_type == "data":
      inputs += [self.data_file]

    # Add other outputs
    inputs += self.other_input_files

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
    extra_dir="",
    extra_name=""
    ):

    import tensorflow as tf

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
        tf.random.set_seed(self.seed)
        tf.keras.utils.set_random_seed(self.seed)
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

      if not self.scale_to_yield:
        sum_sim_hist_total = np.sum(sim_hist_total)
        sum_synth_hist_total = np.sum(synth_hist_total)
        sim_hist_total /= sum_sim_hist_total
        sim_hist_uncert = np.sqrt(sim_hist_uncert_squared)/sum_sim_hist_total
        synth_hist_uncert = np.sqrt(synth_hist_uncert_squared)/sum_synth_hist_total
        for k in synth_hists.keys():
          synth_hists[k] /= sum_synth_hist_total
      else:
        sim_hist_uncert = np.sqrt(sim_hist_uncert_squared)
        synth_hist_uncert = np.sqrt(synth_hist_uncert_squared)       

      plot_stacked_histogram_with_ratio(
        sim_hist_total, 
        synth_hists, 
        bins, 
        data_name = sim_plot_name, 
        xlabel=col,
        ylabel="Events" if self.scale_to_yield else "Density",
        name=f"{self.plots_output}/{extra_dir}generation_{col}{extra_name}", 
        data_errors=sim_hist_uncert, 
        stack_hist_errors=synth_hist_uncert, 
        use_stat_err=False,
        axis_text="",
        )

      plot_histograms_with_ratio(
        [[sim_hists[list(sim_hists.keys())[ind]], synth_hists[list(synth_hists.keys())[ind]]] for ind in range(len(sim_hists.keys()))],
        [[sim_hist_uncerts[list(sim_hists.keys())[ind]], synth_hist_uncerts[list(synth_hists.keys())[ind]]] for ind in range(len(sim_hists.keys()))],
        [[list(sim_hists.keys())[ind], list(synth_hists.keys())[ind]] for ind in range(len(sim_hists.keys()))],
        bins,
        xlabel = col,
        ylabel="Events" if self.scale_to_yield else "Density",
        name=f"{self.plots_output}/{extra_dir}generation_non_stack_same_pad_{col}{extra_name}",    
      )

      plot_many_comparisons(
        sim_hists,
        synth_hists,
        sim_hist_uncerts,
        synth_hist_uncerts,
        bins,
        xlabel = col,
        ylabel="Events" if self.scale_to_yield else "Density",
        name=f"{self.plots_output}/{extra_dir}generation_non_stack_{col}{extra_name}",       
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
          xlabel = col,
          ylabel="Events" if self.scale_to_yield else "Density",
          name=f"{self.plots_output}/{extra_dir}generation_non_stack_inc_total_{col}{extra_name}",       
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
    extra_dir="",
    extra_name=""
    ):

    import tensorflow as tf

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
          tf.random.set_seed(self.seed)
          tf.keras.utils.set_random_seed(self.seed)
          synth_hist, synth_hist_uncert, bins = synth_dps[file_name].GetFull(
            method = "histogram_2d_and_uncert",
            functions_to_apply = functions_to_apply,
            bins = [unrolled_col_bins, plot_col_bins],
            column = [unrolled_col, plot_col],
            )

          tf.random.set_seed(self.seed)
          tf.keras.utils.set_random_seed(self.seed)
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
          name=f"{self.plots_output}/{extra_dir}generation_unrolled_2d_{plot_col}_{unrolled_col}{extra_name}", 
          data_hists_errors=np.sqrt(sim_hist_uncert_squared), 
          stack_hists_errors=np.sqrt(synth_hist_uncert_squared), 
          use_stat_err=False,
          axis_text="",
        )


  def _Plot2DPulls(self, synth_dp, sim_dp):
    print()

  def _PlotCorrelationMatrix(self, synth_dp, sim_dp):
    print()