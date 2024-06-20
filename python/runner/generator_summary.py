import yaml
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from functools import partial

from useful_functions import GetYName
from plotting import plot_histograms

class GeneratorSummary():

  def __init__(self):

    self.parameters = None
    self.model = None
    self.architecture = None

    self.val_loop = None
    self.pois = None
    self.nuisances = None

    self.yield_function = "default"
    self.plots_output = "plots/"
    self.verbose = True
    self.n_synth = 10**6
    self.n_bins = 40
    self.scale_to_yield = False
    self.scale_to_eff_events = False
    self.ratio_to = "synth"
    self.extra_plot_name = ""

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

    # Loop through, make networks and make histograms
    networks = {}
    sim_dps = {}
    synth_dps = {}
    ratio_dps = {}

    sim_hists = {}
    synth_hists = {}
    ratio_hists = {}
    bins = {}
    central_value = {}

    functions_to_apply = ["untransform"]

    # Loop though files
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
          column_name="yield" if not self.scale_to_eff_events else "effective_events"
        )

      # Loop through validation Y values
      for val_ind, val_info in enumerate(self.val_loop):

        val_name = GetYName(val_info["row"], purpose="plot", prefix="y=")

        # Make data processors
        shape_Y_cols = [col for col in val_info["row"].columns if "mu_" not in col and col in parameters["Y_columns"]]

        if self.verbose:
          print(f"- Making data processor for {file_name} and {val_name}")

        sim_dps[file_name] = DataProcessor(
          [[f"{parameters['file_loc']}/X_val.parquet", f"{parameters['file_loc']}/Y_val.parquet", f"{parameters['file_loc']}/wt_val.parquet"]],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : parameters,
            "selection" : " & ".join([f"({col}=={val_info['row'].loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
            "scale" : yields.GetYield(val_info["row"]) if self.scale_to_yield else 1.0,
          }
        )

        synth_dps[file_name] = DataProcessor(
          [[partial(networks[file_name].Sample, val_info["row"])]],
          "generator",
          n_events = self.n_synth,
          options = {
            "parameters" : parameters,
            "scale" : yields.GetYield(val_info["row"]) if self.scale_to_yield else 1.0,
          }
        )

        if val_ind == 0:
          if self.ratio_to == "synth":
            ratio_dps[file_name] = DataProcessor(
              [[partial(networks[file_name].Sample, val_info["initial_best_fit_guess"])]],
              "generator",
              n_events = self.n_synth,
              options = {
                "parameters" : parameters,
                "scale" : yields.GetYield(val_info["initial_best_fit_guess"]) if self.scale_to_yield else 1.0,
              }
            )
          elif self.ratio_to == "sim":
            ratio_dps[file_name] = DataProcessor(
              [[f"{parameters['file_loc']}/X_val.parquet", f"{parameters['file_loc']}/Y_val.parquet", f"{parameters['file_loc']}/wt_val.parquet"]],
              "parquet",
              wt_name = "wt",
              options = {
                "parameters" : parameters,
                "selection" : " & ".join([f"({col}=={val_info['initial_best_fit_guess'].loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
                "scale" : yields.GetYield(val_info["initial_best_fit_guess"]) if self.scale_to_yield else 1.0,
              }
            )

        # Loop through columns
        for col in parameters["X_columns"]:

          if self.verbose:
            print(f"- Drawing histograms for {file_name}, {val_name} and {col}")

          # Make bins and find central value
          if col not in bins.keys():

            bins[col] = ratio_dps[file_name].GetFull(
              method = "bins_with_equal_spacing", 
              functions_to_apply = functions_to_apply,
              bins = self.n_bins,
              column = col,
            )

          # Draw ratio histogram
          if val_ind == 0:
            ratio_hist, _ = ratio_dps[file_name].GetFull(
              method = "histogram",
              functions_to_apply = functions_to_apply,
              bins = bins[col],
              column = col,
              )
            if col not in ratio_hists.keys():
              ratio_hists[col] = copy.deepcopy(ratio_hist)
            else:
              ratio_hists[col] += ratio_hist

          # Draw synth histogram
          synth_hist, _ = synth_dps[file_name].GetFull(
            method = "histogram",
            functions_to_apply = functions_to_apply,
            bins = bins[col],
            column = col,
            )
          if col not in synth_hists.keys():
            synth_hists[col] = {val_name : copy.deepcopy(synth_hist)}
          else:
            if val_name in synth_hists[col].keys():
              synth_hists[col][val_name] += synth_hist
            else:
              synth_hists[col][val_name] = synth_hist

          # Draw sim_histogram
          sim_hist, _ = sim_dps[file_name].GetFull(
            method = "histogram",
            functions_to_apply = functions_to_apply,
            bins = bins[col],
            column = col,
            )
          if col not in sim_hists.keys():
            sim_hists[col] = {val_name : copy.deepcopy(sim_hist)}
          else:
            if val_name in sim_hists[col].keys():
              sim_hists[col][val_name] += sim_hist
            else:
              sim_hists[col][val_name] = sim_hist

    # Loop through columns
    for col in parameters["X_columns"]:

      # Make ratios
      for key, val in sim_hists[col].items():
        sim_hists[col][key] /= ratio_hists[col]
      for key, val in synth_hists[col].items():
        synth_hists[col][key] /= ratio_hists[col]   
      
      # Get colours, linestyles and histograms ordered
      rgb_palette = sns.color_palette("Set2", len(list(sim_hists[col].keys())))
      colours = ["black","black"]
      names = [
        GetYName(val_info["initial_best_fit_guess"], purpose="plot", prefix="Synthetic (y=")+")",
        GetYName(val_info["initial_best_fit_guess"], purpose="plot", prefix="Simulated (y=")+")"
        ]
      hists = [
        synth_hists[col][GetYName(val_info["initial_best_fit_guess"], purpose="plot", prefix="y=")],
        sim_hists[col][GetYName(val_info["initial_best_fit_guess"], purpose="plot", prefix="y=")]
      ]
      linestyles = ["-", "--"]
      for ind in range(len(list(sim_hists[col].keys()))-1):
        colour = tuple(x for x in rgb_palette[ind])
        colours.append(colour)
        names.append(list(synth_hists[col].keys())[ind+1])
        hists.append(synth_hists[col][names[-1]])
        linestyles.append("-")

        if names[-1] in sim_hists[col].keys():
          colours.append(colour)
          hists.append(sim_hists[col][names[-1]])
          names.append(None)
          linestyles.append("--")

      # Set up y axis name
      if self.ratio_to == "synth":
        y_label = f"Ratio to {GetYName(val_info['initial_best_fit_guess'], purpose='plot', prefix='Synthetic y=')}"
      elif self.ratio_to == "sim":
        y_label = f"Ratio to {GetYName(val_info['initial_best_fit_guess'], purpose='plot', prefix='Simulated y=')}"

      # Make plot
      plot_histograms(
        bins[col][:-1],
        hists,
        names,
        colors = colours,
        linestyles = linestyles,
        x_label=col,
        name=f"{self.plots_output}/GenerationSummary/generation_summary_{col}{self.extra_plot_name}", 
        y_label = y_label
      )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    file_name = list(self.model.keys())[0]
    with open(self.parameters[file_name], 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    for col in parameters["X_columns"]:
      outputs += [f"{self.plots_output}/GenerationSummary/generation_summary_{col}{self.extra_plot_name}.pdf"]

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
        f"{parameters['file_loc']}/X_train.parquet",
        f"{parameters['file_loc']}/Y_train.parquet", 
        f"{parameters['file_loc']}/wt_train.parquet", 
        f"{parameters['file_loc']}/X_test.parquet",
        f"{parameters['file_loc']}/Y_test.parquet", 
        f"{parameters['file_loc']}/wt_test.parquet",        
      ]
    return inputs