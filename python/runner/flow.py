import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml

import numpy as np
import pandas as pd

from functools import partial

from plotting import plot_histograms
from useful_functions import GetYName

class Flow():

  def __init__(self):

    self.parameters = None
    self.model = None
    self.architecture = None
    self.Y_sim = None

    self.plots_output = "plots/"
    self.verbose = True
    self.data_type = "sim"
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

    # Loop through and make networks
    networks = {}
    sim_dps = {}
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
          "functions" : ["untransform","transform"]
        }
      )

      # Make bins
      bins = np.linspace(-3,3,num=100)

      # Loop through columns
      for col in range(max(len(parameters["X_columns"]),2)):

        # Make nominal histogram and plot
        if not (col == 1 and len(parameters["X_columns"]) == 1):
          hist, _ = sim_dps[file_name].GetFull(method="histogram", bins=bins, column=parameters["X_columns"][col], density=True)
          plot_histograms(
            bins[:-1],
            [hist],
            [None],
            name = f"{self.plots_output}/flow_{col}_cl0",
            x_label = col,
            y_label = "Density",
          )

        num_coupling_layers = len(networks[parameters['file_name']].inference_net.coupling_layers)
        for ind in range(num_coupling_layers):
          flow_func = partial(self._FlowTransform, network=networks[parameters['file_name']], coupling_layer=ind, X_columns=parameters["X_columns"], Y_columns=parameters["Y_columns"])
          hist, _ = sim_dps[file_name].GetFull(method="histogram", bins=bins, column=f"FlowTransform_{col}", density=True, functions_to_apply=[flow_func])
          plot_histograms(
            bins[:-1],
            [hist],
            [None],
            name = f"{self.plots_output}/flow_{col}_cl{ind+1}",
            x_label = col,
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
    outputs = []
    file_name = list(self.model.keys())[0]
    with open(self.parameters[file_name], 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Add other outputs
    outputs += []

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

    return inputs