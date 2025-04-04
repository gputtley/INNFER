import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from scipy.interpolate import CubicSpline

from data_processor import DataProcessor
from useful_functions import InitiateDensityModel, InitiateRegressionModel, MakeDirectories, LoadConfig, GetDefaultsInModel
from yields import Yields

class MakeAsimov():

  def __init__(self):

    self.cfg = None

    self.file_name = None
    self.density_model = None
    self.regression_models = None
    self.model_extra_name = ""
    self.parameters = None
    self.model_input = "data/"
    self.data_output = "data/"
    self.regression_spline_input = "data/"
    self.n_asimov_events = 10**7
    self.seed = 42
    self.val_info = {}
    self.val_ind = None
    self.only_density = False
    self.Y = None
    self.add_truth = False
    self.verbose = True

  def _WriteDataset(self, df, file_name):

    file_path = f"{self.data_output}/{file_name}"
    MakeDirectories(file_path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(file_path):
      combined_table = pa.concat_tables([pq.read_table(file_path), table])
      pq.write_table(combined_table, file_path, compression='snappy')
    else:
      pq.write_table(table, file_path, compression='snappy')

    return df


  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)


  def Run(self):

    # Import costly packages
    import tensorflow as tf
    
    # Open cfg
    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg)

    # Open parameters
    if self.verbose:
      print("- Loading in parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Make all parameters
    if self.verbose:
      print("- Find model parameters")
    model_parameters = GetDefaultsInModel(parameters["file_name"], cfg)
    for lnN in parameters["yields"]["lnN"].keys():
      model_parameters[lnN] = 0.0
    for key, val in self.val_info.items():
      model_parameters[key] = val

    # Find yield
    if self.verbose:
      print("- Calculate yields prediction")
    yield_class = Yields(
      parameters["yields"]["nominal"],
      lnN = parameters["yields"]["lnN"],
      physics_model = None,
      rate_param = f"mu_{parameters['file_name']}" if f"mu_{parameters['file_name']}" in model_parameters.keys() else None,
    )
    total_yield = yield_class.GetYield(pd.DataFrame({k:[v] for k,v in model_parameters.items()}))

    # Build the density model
    if self.verbose:
      print("- Building density network")

    density_model_name = f"{self.model_input}/{self.density_model['name']}/{parameters['file_name']}"
    with open(f"{density_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    network = InitiateDensityModel(
      architecture,
      self.density_model['file_loc'],
      options = {
        "data_parameters" : parameters["density"],
      }
    )
  
    # Loading density model
    if self.verbose:
      print(f"- Loading the density model {density_model_name}{self.model_extra_name}")
    network.Load(name=f"{density_model_name}{self.model_extra_name}.h5")

    # Sample from density model
    if self.verbose:
      print(f"- Sampling from density network")
    if self.Y is not None:
      Y = self.Y
    else:
      Y = pd.DataFrame({k:[v] for k,v in model_parameters.items() if k in parameters["density"]["Y_columns"]})
    asimov_writer = DataProcessor(
      [[partial(network.Sample, Y)]],
      "generator",
      n_events = self.n_asimov_events,
      wt_name = "wt",
      options = {
        "parameters" : parameters["density"],
        "scale" : total_yield,
      }
    )
    def add_truth(df, Y):
      for k,v in Y.items():
        df.loc[:,k] = v
      return df

    functions_to_apply = []
    if self.add_truth:
      functions_to_apply += [partial(add_truth, Y=model_parameters)]
    functions_to_apply += [partial(self._WriteDataset,file_name="asimov.parquet")]


    asimov_file_name = f"{self.data_output}/asimov.parquet"
    MakeDirectories(asimov_file_name)
    if os.path.isfile(asimov_file_name): os.system(f"rm {asimov_file_name}")
    tf.random.set_seed(self.seed)
    tf.keras.utils.set_random_seed(self.seed)
    asimov_writer.GetFull(
      method = None,
      functions_to_apply = functions_to_apply
    )

    if not self.only_density:

      # Do regression models
      for regression_model in self.regression_models:

        # Build the regression model
        if self.verbose:
          print(f"- Building regresson network for {regression_model['parameter']}")
        regression_model_name = f"{self.model_input}/{regression_model['name']}/{parameters['file_name']}"
        with open(f"{regression_model_name}_architecture.yaml", 'r') as yaml_file:
          architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

        network = InitiateRegressionModel(
          architecture,
          regression_model['file_loc'],
          options = {
            "data_parameters" : parameters['regression'][regression_model['parameter']]    
          }
        )  
      
        # Loading density model
        if self.verbose:
          print(f"- Loading the density model {regression_model_name}")
        network.Load(name=f"{regression_model_name}.h5")

        # Apply weights
        wt_shifter = DataProcessor(
          [[asimov_file_name]],
          "parquet",
          wt_name = "wt",
          options = {
          }
        )

        # Open normalising spline
        if self.verbose:
          print(f"- Loading the normalising spline for {regression_model_name}")

        spline_name = f"{regression_model_name}_norm_spline.pkl"
        if os.path.isfile(spline_name):
          with open(spline_name, 'rb') as f:
              spl = pickle.load(f)
        else:
          print(f"WARNING: No normalising spline found for {regression_model['parameter']}. Leaving unnormalised.")
          spl = None

        def apply_regression(df, func, X_columns, add_columns={}, spl=None, parameter=None):
          cols_in = list(df.columns)
          for k,v in add_columns.items(): df.loc[:,k] = v
          df.loc[:,"wt"] *= func(df.loc[:,X_columns]).flatten()
          if spl is not None:
            df.loc[:,"wt"] *= spl(df.loc[:,parameter]).flatten()
          return df.loc[:,cols_in]

        wt_shifter_name = asimov_file_name.replace('.parquet','_wt_shifter.parquet')
        if os.path.isfile(wt_shifter_name): os.system(f"rm {wt_shifter_name}")

        wt_shifter.GetFull(
          method = None,
          functions_to_apply = [
            partial(
              apply_regression, 
              func=network.Predict, 
              X_columns=parameters['regression'][regression_model['parameter']]["X_columns"],
              add_columns={regression_model['parameter']: model_parameters[regression_model['parameter']]},
              spl = spl,
              parameter = regression_model['parameter']
            ),
            partial(self._WriteDataset,file_name=wt_shifter_name.split("/")[-1])
          ]
        )

        if os.path.isfile(wt_shifter_name): os.system(f"mv {wt_shifter_name} {asimov_file_name}")

  def Outputs(self):

    # Add asimov
    outputs = [f"{self.data_output}/asimov.parquet"]

    return outputs

  def Inputs(self):

    # Initiate inputs
    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add parameters
    inputs += [self.parameters]

    # Add density model
    inputs += [f"{self.model_input}/{self.density_model['name']}/{self.file_name}_architecture.yaml"]
    inputs += [f"{self.model_input}/{self.density_model['name']}/{self.file_name}.h5"]

    # Add regression models
    for regression_model in self.regression_models:
      inputs += [f"{self.model_input}/{regression_model['name']}/{self.file_name}_architecture.yaml"]
      inputs += [f"{self.model_input}/{regression_model['name']}/{self.file_name}.h5"]
      inputs += [f"{self.model_input}/{regression_model['name']}/{self.file_name}_norm_spline.pkl"]

    return inputs