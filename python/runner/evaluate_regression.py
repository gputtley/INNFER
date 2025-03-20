import copy
import os
import pickle
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from scipy.interpolate import CubicSpline

from data_processor import DataProcessor
from useful_functions import InitiateRegressionModel, MakeDirectories

class EvaluateRegression():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None

    # other
    self.data_output = None
    self.model_input = None
    self.model_name = None
    self.parameter = None
    self.verbose = True
    self.test_name = "test"
    self.model_type = "FCNN"

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
    """
    Run the code utilising the worker classes
    """

    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the model in
    if self.verbose:
      print("- Building the model")
    regression_model_name = f"{self.model_input}/{self.model_name}/{parameters['file_name']}"
    with open(f"{regression_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
    network = InitiateRegressionModel(
      architecture,
      parameters["regression"][self.parameter]['file_loc'],
      options = {
        "data_parameters" : parameters['regression'][self.parameter]    
      }
    )  
    network.BuildModel()
    network.BuildTrainer()
    network.Load(name=f"{regression_model_name}.h5")

    # Make y from train and test
    loop = ["train"]
    if self.test_name is not None:
      loop.append(self.test_name)

    for data_split in loop:
      if self.verbose:
        print(f"- Getting to the predictions for the {data_split} dataset")

      pred_df = DataProcessor(
        [f"{parameters['regression'][self.parameter]['file_loc']}/X_{data_split}.parquet"],
        "parquet",
        options = {
          "parameters" : parameters['regression'][self.parameter]
        }
      )


      def apply_regression(df, func, X_columns):
        df.loc[:,"wt_shift"] = func(df.loc[:,X_columns])
        return df.loc[:,["wt_shift"]]

      pred_name = f"{self.data_output}/pred_{data_split}.parquet"
      if os.path.isfile(pred_name): os.system(f"rm {pred_name}")

      pred_df.GetFull(
        method = None,
        functions_to_apply = [
          "untransform",
          partial(
            apply_regression, 
            func=network.Predict, 
            X_columns=parameters['regression'][self.parameter]["X_columns"],
          ),
          partial(self._WriteDataset,file_name=f"pred_{data_split}.parquet")
        ]
      )


      # Get normalisation spline
      if data_split == "train":

        if self.verbose:
          print("- Getting normalisation spline")

        parameters_removed_standardisation = copy.deepcopy(parameters['regression'][self.parameter])
        if "standardisation" in parameters_removed_standardisation.keys():
          if "wt_shift" in parameters_removed_standardisation["standardisation"].keys():
            del parameters_removed_standardisation["standardisation"]["wt_shift"]

        regress_df = DataProcessor(
          [f"{parameters_removed_standardisation['file_loc']}/X_{data_split}.parquet", f"{parameters_removed_standardisation['file_loc']}/wt_{data_split}.parquet", pred_name],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : parameters_removed_standardisation,
            "functions" : ["untransform"],
          }
        )

        nom_hist, bins = regress_df.GetFull(
          method = "histogram",
          column = self.parameter,
          bins = 40,
        )

        def shift(df):
          df.loc[:,"wt"] *= df.loc[:,"wt_shift"]
          return df

        shifted_hist, _ = regress_df.GetFull(
          method = "histogram",
          bins = bins,
          column = self.parameter,
          functions_to_apply = [shift]
        )
        
        ratio = nom_hist/shifted_hist
        bin_centers = (bins[:-1] + bins[1:]) / 2
        spline = CubicSpline(bin_centers, ratio, bc_type="clamped", extrapolate=False)
        spline_name = f"{regression_model_name}_norm_spline.pkl"
        with open(spline_name, 'wb') as f:
          pickle.dump(spline, f)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Add output data
    outputs += [f"{self.data_output}/pred_train.parquet"]
    if self.test_name is not None:
      outputs += [f"{self.data_output}/pred_{self.test_name}.parquet"]

    # Add normalisation spline
    outputs += [f"{self.model_input}/{self.model_name}/{parameters['file_name']}_norm_spline.pkl"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initiate inputs
    inputs = []

    # Add parameters
    inputs += [self.parameters]

    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Add model
    inputs += [f"{self.model_input}/{self.model_name}/{parameters['file_name']}_architecture.yaml"]
    inputs += [f"{self.model_input}/{self.model_name}/{parameters['file_name']}.h5"]

    # Add data
    inputs += [
      f"{parameters['regression'][self.parameter]['file_loc']}/X_train.parquet", 
      f"{parameters['regression'][self.parameter]['file_loc']}/wt_train.parquet",
    ]
    if self.test_name is not None:
      inputs += [
        f"{parameters['regression'][self.parameter]['file_loc']}/X_{self.test_name}.parquet",
        f"{parameters['regression'][self.parameter]['file_loc']}/wt_{self.test_name}.parquet",
      ]

    return inputs

        