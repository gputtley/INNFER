import yaml
import copy
import numpy as np
from functools import partial

from useful_functions import GetYName
from plotting import plot_stacked_histogram_with_ratio

class Generator():

  def __init__(self):

    self.parameters = None
    self.model = None
    self.architecture = None
    self.Y_sim = None
    self.Y_synth = None

    self.yield_function = "default"
    self.plots_output = "plots/"
    self.verbose = True
    self.n_synth = 10**6

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):

    from data_processor import DataProcessor
    from network import Network

    # Make singular inputs as dictionaries
    combined_model = True
    if isinstance(self.model, str):
      combined_model = False
      with open(self.parameters, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
      self.model = {parameters['file_name'] : self.model}
      self.parameters = {parameters['file_name'] : self.parameters}
      self.architecture = {parameters['file_name'] : self.architecture}

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

      # Make data processors
      if self.verbose:
        print(f"- Making data processor for {file_name}")
      sim_dps[file_name] = DataProcessor(
        [[f"{parameters['file_loc']}/X_val.parquet", f"{parameters['file_loc']}/Y_val.parquet", f"{parameters['file_loc']}/wt_val.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : parameters,
          "selection" : " & ".join([f"({list(self.Y_sim.columns)[ind]}=={y_sim})" for ind, y_sim in enumerate(self.Y_sim.to_numpy().flatten())])
        }
      )
      synth_dps[file_name] = DataProcessor(
        [[partial(networks[file_name].Sample, self.Y_synth)]],
        "generator",
        n_events = self.n_synth,
        options = {
          "parameters" : parameters,
        }
      )

    # Get names for plot
    sim_file_name = GetYName(self.Y_sim.to_numpy(), purpose="file", prefix="_sim_y_")
    sample_file_name = GetYName(self.Y_synth.to_numpy(), purpose="file", prefix="_synth_y_")
    sim_plot_name = GetYName(self.Y_sim.to_numpy(), purpose="plot", prefix="Simulated y=")
    sample_plot_name = GetYName(self.Y_synth.to_numpy(), purpose="plot", prefix="Synthetic y=")


    if self.verbose:
      print(f"- Making 1D generation plots")
    # Running plotting functions
    self._PlotGeneration(
      synth_dps, 
      synth_dps, 
      parameters["X_columns"],
      sim_file_name,
      sample_file_name,
      sim_plot_name,
      sample_plot_name
    )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    return inputs

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
    transform=False
    ):

    functions_to_apply = []
    if not transform:
      functions_to_apply.append("untransform")

    synth_hists = {}

    for col in X_columns:

      bins = copy.deepcopy(n_bins)

      for ind, file_name in enumerate(synth_dps.keys()):

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
          sim_hist_uncert_squared = synth_hist_uncert**2
          sim_hist_total = copy.deepcopy(sim_hist)
        else:
          synth_hist_uncert_squared += (synth_hist_uncert**2)
          sim_hist_uncert_squared += (synth_hist_uncert**2)
          sim_hist_total += sim_hist

        synth_hists[f"{file_name} {synth_plot_name}"] = synth_hist

      plot_stacked_histogram_with_ratio(
        sim_hist_total, 
        synth_hists, 
        bins, 
        data_name = sim_plot_name, 
        xlabel=col,
        ylabel="Events",
        name=f"{self.plots_output}/generation_{col}{sim_file_name}{synth_file_name}", 
        data_errors=np.sqrt(sim_hist_uncert_squared), 
        stack_hist_errors=np.sqrt(synth_hist_uncert_squared), 
        title_right="",
        use_stat_err=False,
        axis_text="",
        )

  def _Plot2DUnrolledGeneration(self, synth_dp, sim_dp):
    print()

  def _Plot2DPulls(self, synth_dp, sim_dp):
    print()

  def _PlotCorrelationMatrix(self, synth_dp, sim_dp):
    print()