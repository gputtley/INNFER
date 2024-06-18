import yaml
import pandas as pd
from functools import partial

from data_processor import DataProcessor
from likelihood import Likelihood
from yields import Yields
from network import Network

class Infer():

  def __init__(self):

    self.parameters =  None
    self.model = None
    self.architecture = None

    self.true_Y = None
    self.initial_best_fit_guess = None
    self.pois = None
    self.nuisances = None
    self.method = "InitialFit"

    self.inference_options = {}
    self.yield_function = "default",
    self.data_type = "sim"
    self.data_output = "data"
    self.likelihood_type = "unbinned_extended"
    self.resample = False
    self.resampling_seed = 1
    self.verbose = True
    self.n_asimov_events = 10**6

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

    lkld = self._BuildLikelihood()
    dps = self._BuildDataProcessors(lkld)

    if self.method == "InitialFit":

      if self.verbose:
        if self.data_type != "data":
          print(f"- Performing initial fit for the dataframe on data type {self.data_type}:")
          print(self.true_Y)
        else:
          print(f"- Performing initial fit on data")

      for v in [167.0,168.0,169.0,170.0,171.0,172.0,173.0]:
        lkld.Run(dps.values(), pd.DataFrame([[v]], columns=["true_mass"]))



  def Outputs(self):
    outputs = []
    return outputs

  def Inputs(self):
    inputs = []
    return inputs

  def _BuildDataProcessors(self, lkld):

    dps = {}
    for file_name in self.model.keys():

      if self.data_type == "sim":

        shape_Y_cols = [col for col in self.true_Y.columns if "mu_" not in col and col in lkld.data_parameters[file_name]["Y_columns"]]
        dps[file_name] = DataProcessor(
          [[f"{lkld.data_parameters[file_name]['file_loc']}/X_val.parquet", f"{lkld.data_parameters[file_name]['file_loc']}/Y_val.parquet", f"{lkld.data_parameters[file_name]['file_loc']}/wt_val.parquet"]],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : lkld.data_parameters[file_name],
            "selection" : " & ".join([f"({col}=={self.true_Y.loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
            "scale" : lkld.models["yields"][file_name](self.true_Y),
          }
        )

      elif self.data_type == "asimov":

        dps[file_name] = DataProcessor(
          [[partial(lkld.models["pdfs"][file_name].Sample, self.true_Y)]],
          "generator",
          n_events = self.n_asimov_events,
          options = {
            "parameters" : lkld.data_parameters[file_name],
            "scale" : lkld.models["yields"][file_name](self.true_Y),
          }
        )

      elif self.data_type == "data":

        print("Still need to implement data")

      return dps


  def _BuildYieldFunctions(self):

    yields = {}
    for k, v in self.parameters.items():

      with open(v, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

      yields_class = Yields(
        pd.read_parquet(parameters['yield_loc']), 
        self.pois, 
        self.nuisances, 
        k,
        method=self.yield_function, 
        column_name="yield"
      )
      yields[k] = yields_class.GetYield

    return yields

  def _BuildModels(self):

    networks = {}
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

    return networks

  def _BuildLikelihood(self):

    if self.verbose:
      print(f"- Building likelihood")

    parameters = {}
    for file_name in self.model.keys():
      with open(self.parameters[file_name], 'r') as yaml_file:
        parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    lkld = Likelihood(
      {
        "pdfs" : self._BuildModels(),
        "yields" : self._BuildYieldFunctions()
      }, 
      likelihood_type = self.likelihood_type, 
      data_parameters = parameters,
      parameters = self.inference_options,
    )

    return lkld

