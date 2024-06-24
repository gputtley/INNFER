import yaml
import pandas as pd
import numpy as np
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
    self.data_input = "data"
    self.data_output = "data"
    self.likelihood_type = "unbinned_extended"
    self.resample = False
    self.resampling_seed = 1
    self.verbose = True
    self.n_asimov_events = 10**6
    self.minimisation_method = "nominal"
    self.freeze = {}
    self.extra_file_name = ""
    self.scale_to_eff_events = False
    self.column = None
    self.scan_value = None
    self.scan_ind = None
    self.sigma_between_scan_points = 0.4
    self.number_of_scan_points = 17
    self.other_input = None

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

    if self.extra_file_name != "":
      self.extra_file_name = f"_{self.extra_file_name}"


  def Run(self):

    lkld = self._BuildLikelihood()
    dps = self._BuildDataProcessors(lkld)

    if self.verbose:
      if self.data_type != "data":
        print(f"- Performing actions on the likelihood for the dataframe on data type {self.data_type}:")
        print(self.true_Y)
      else:
        print(f"- Performing actions on the likelihood on data")

    if self.method == "InitialFit":

      if self.verbose:
        print("- Likelihood output:")

      lkld.GetAndWriteBestFitToYaml(
        dps.values(), 
        self.initial_best_fit_guess, 
        row=self.true_Y, 
        filename=f"{self.data_output}/best_fit{self.extra_file_name}.yaml", 
        minimisation_method=self.minimisation_method, 
        freeze=self.freeze,
      )

    elif self.method == "ScanPoints":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.data_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      lkld.best_fit = np.array(best_fit_info["best_fit"])
      lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Finding values to perform the scan")
        print("- Likelihood output:")

      # Make scan ranges
      lkld.GetAndWriteScanRangesToYaml(
        dps.values(), 
        self.column,
        row=self.true_Y,
        estimated_sigmas_shown=((self.number_of_scan_points-1)/2)*self.sigma_between_scan_points, 
        estimated_sigma_step=self.sigma_between_scan_points,
        filename=f"{self.data_output}/scan_ranges_{self.column}{self.extra_file_name}.yaml"
      )

    elif self.method == "Scan":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.data_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      lkld.best_fit = np.array(best_fit_info["best_fit"])
      lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Performing likelihood profiling")
        print(f"- Likelihood output:")

      lkld.GetAndWriteScanToYaml(
        dps.values(), 
        self.column, 
        self.scan_value, 
        row=self.true_Y, 
        freeze=self.freeze, 
        filename=f"{self.data_output}/scan_values_{self.column}{self.extra_file_name}_{self.scan_ind}.yaml", 
        minimisation_method=self.minimisation_method
      )

    elif self.method == "Debug":

      if self.verbose:
        print(f"- Likelihood output:")

      for j in self.other_input.split(":"):
        lkld.Run(dps.values(), [float(i) for i in j.split(',')], Y_columns=list(self.initial_best_fit_guess.columns))

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

        if lkld.models["yields"][file_name](self.true_Y) == 0: continue
        shape_Y_cols = [col for col in self.true_Y.columns if "mu_" not in col and col in lkld.data_parameters[file_name]["Y_columns"]]
        dps[file_name] = DataProcessor(
          [[f"{lkld.data_parameters[file_name]['file_loc']}/X_val.parquet", f"{lkld.data_parameters[file_name]['file_loc']}/Y_val.parquet", f"{lkld.data_parameters[file_name]['file_loc']}/wt_val.parquet"]],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : lkld.data_parameters[file_name],
            "selection" : " & ".join([f"({col}=={self.true_Y.loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
            "scale" : lkld.models["yields"][file_name](self.true_Y),
            "resample" : self.resample,
            "resampling_seed" : self.resampling_seed,
          }
        )

      elif self.data_type == "asimov":

        if lkld.models["yields"][file_name](self.true_Y) == 0: continue
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
    parameters = {}
    for k, v in self.parameters.items():

      with open(v, 'r') as yaml_file:
        parameters[k] = yaml.load(yaml_file, Loader=yaml.FullLoader)

      yields_class = Yields(
        pd.read_parquet(parameters[k]['yield_loc']), 
        self.pois, 
        self.nuisances, 
        k,
        method=self.yield_function, 
        column_name="yield" if not self.scale_to_eff_events else "effective_events"
      )
      yields[k] = yields_class.GetYield

    return yields

  def _BuildModels(self):

    networks = {}
    parameters = {}
    for file_name in self.model.keys():

      # Open parameters
      if self.verbose:
        print(f"- Loading in the parameters for model {file_name}")
      with open(self.parameters[file_name], 'r') as yaml_file:
        parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Load the architecture in
      if self.verbose:
        print(f"- Loading in the architecture for model {file_name}")
      with open(self.architecture[file_name], 'r') as yaml_file:
        architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Build model
      if self.verbose:
        print(f"- Building the model for {file_name}")
      networks[file_name] = Network(
        f"{parameters[file_name]['file_loc']}/X_train.parquet",
        f"{parameters[file_name]['file_loc']}/Y_train.parquet", 
        f"{parameters[file_name]['file_loc']}/wt_train.parquet", 
        f"{parameters[file_name]['file_loc']}/X_test.parquet",
        f"{parameters[file_name]['file_loc']}/Y_test.parquet", 
        f"{parameters[file_name]['file_loc']}/wt_test.parquet",
        options = {
          **architecture,
          **{
            "data_parameters" : parameters[file_name]
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

