import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml

import numpy as np
import pandas as pd

from functools import partial

from plotting import plot_histograms
from useful_functions import InitiateDensityModel, Translate

class Flow():

  def __init__(self):

    self.density_model = None
    self.data_input = None

    self.plots_output = "plots/"
    self.verbose = True
    self.sim_type = "val"
    self.extra_plot_name = ""


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

    from data_processor import DataProcessor

    # Open parameters
    if self.verbose:
      print("- Loading in parameters")

    with open(self.density_model["parameters"], 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build the density model
    if self.verbose:
      print("- Building density network")

    density_model_name = f"{self.model_input}/{self.density_model['name']}/{parameters['file_name']}"
    with open(f"{density_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if architecture["type"] != "BayesFlow":
      raise ValueError("Flow only works for a BayesFlow model.")

    network = InitiateDensityModel(
      architecture,
      self.density_model['file_loc'],
      options = {
        "data_parameters" : parameters["density"],
      }
    )
    network.Load(name=f"{density_model_name}.h5")

    sim_dps = DataProcessor(
      [[f"{self.data_input}/{i}_{self.sim_type}.parquet" for i in ["X","Y","wt"]]],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : parameters["density"],
        "functions" : ["transform"]
      }
    )

    # Make bins
    bins = np.linspace(-3,3,num=100)

    # Loop through columns
    for col in range(len(parameters["density"]["X_columns"])):

      # Make nominal histogram and plot
      hist, _ = sim_dps.GetFull(method="histogram", bins=bins, column=parameters["density"]["X_columns"][col], density=True)
      plot_histograms(
        bins[:-1],
        [hist],
        [None],
        name = f"{self.plots_output}/flow_{col}_cl0{self.extra_plot_name}",
        x_label = "",
        y_label = "Density",
      )

      num_coupling_layers = architecture["num_coupling_layers"]
      for ind in range(num_coupling_layers):
        flow_func = partial(self._FlowTransform, network=network, coupling_layer=ind, X_columns=parameters["density"]["X_columns"], Y_columns=parameters["density"]["Y_columns"])
        hist, _ = sim_dps.GetFull(method="histogram", bins=bins, column=f"FlowTransform_{col}", density=True, functions_to_apply=[flow_func])
        plot_histograms(
          bins[:-1],
          [hist],
          [None],
          name = f"{self.plots_output}/flow_{col}_cl{ind+1}{self.extra_plot_name}",
          x_label = "",
          y_label = "Density",
        )

        
  def _FlowTransform(self, df, network, coupling_layer, X_columns, Y_columns):

    coupling_layers = network.inference_net.coupling_layers[:]
    network.inference_net.coupling_layers = coupling_layers[:coupling_layer+1]
    z, _ = network.inference_net(df.loc[:,X_columns], df.loc[:,Y_columns])
    network.inference_net.coupling_layers = coupling_layers[:]

    # Make dataframe
    df.loc[:,[f"FlowTransform_{ind}" for ind in range(z.numpy().shape[1])]] = z.numpy()

    return df


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Open parameters
    with open(self.density_model["parameters"], 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    density_model_name = f"{self.model_input}/{self.density_model['name']}/{parameters['file_name']}"
    with open(f"{density_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
    num_coupling_layers = architecture["num_coupling_layers"]


    # Add plots
    for col in range(len(parameters["density"]["X_columns"])):
      outputs += [f"{self.plots_output}/flow_{col}_cl0{self.extra_plot_name}.pdf"]
      for ind in range(num_coupling_layers):
        outputs += [f"{self.plots_output}/flow_{col}_cl{ind+1}{self.extra_plot_name}.pdf"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initiate inputs
    inputs = []

    # Add parameters
    inputs += [self.density_model["parameters"]]

    # Open parameters
    with open(self.density_model["parameters"], 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Add model
    inputs += [f"{self.model_input}/{self.density_model['name']}/{parameters['file_name']}_architecture.yaml"]
    inputs += [f"{self.model_input}/{self.density_model['name']}/{parameters['file_name']}.h5"]

    # Add data
    inputs += [f"{self.data_input}/{i}_{self.sim_type}.parquet" for i in ["X","Y","wt"]]

    return inputs