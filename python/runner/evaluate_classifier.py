import copy
import os
import pickle
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from functools import partial
#from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline


from data_processor import DataProcessor
from make_asimov import MakeAsimov
from plotting import plot_histograms
from useful_functions import GetDefaultsInModel, GetModelLoop, InitiateClassifierModel, LoadConfig, MakeDirectories

class EvaluateClassifier():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None
    self.cfg = None

    # other
    self.data_input = "data/"
    self.file_name = None
    self.data_output = None
    self.model_input = None
    self.model_name = None
    self.parameter = None
    self.verbose = True
    self.get_auc = True
    self.test_name = "test"
    self.model_type = "FCNN"
    self.spline_from_asimov = True
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    self.extra_classifier_model_name = ""
  

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

    if self.extra_classifier_model_name != "":
      self.extra_classifier_model_name = f"_{self.extra_classifier_model_name}"


  def Run(self):
    """
    Run the code utilising the worker classes
    """

    # Define loop
    loop = []

    if self.get_auc:
      performance_metrics = {}

    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the model in
    if self.verbose:
      print("- Building the model")
    classifier_model_name = f"{self.model_input}/{self.model_name}/{parameters['file_name']}{self.extra_classifier_model_name}"
    with open(f"{classifier_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
    network = InitiateClassifierModel(
      architecture,
      self.data_input,
      options = {
        "data_parameters" : parameters['classifier'][self.parameter]
      }
    )  
    network.BuildModel()
    network.BuildTrainer()
    network.Load(name=f"{classifier_model_name}.h5")

    # Make y from train and test
    loop.append("train")
    if self.test_name is not None:
      loop.append(self.test_name)

    for data_split in loop:
      if self.verbose:
        print(f"- Getting to the predictions for the {data_split} dataset")

      files = [f"{parameters['classifier'][self.parameter]['file_loc']}/{i}_{data_split}.parquet" for i in ["X","y","wt"]]

      pred_df = DataProcessor(
        [files],
        "parquet",
        batch_size = self.batch_size,
        options = {
          "parameters" : parameters['classifier'][self.parameter],
        },
      )

      def apply_classifier(df, func, X_columns):
        df.loc[:,"wt_shift"] = 1.0
        probs = func(df.loc[:,X_columns])
        inds = (df["classifier_truth"] == 0)
        df.loc[inds, "wt_shift"] = probs[inds,1] / probs[inds, 0]
        df.loc[:, "probs"] = probs[:,1]
        return df.loc[:,["wt_shift","probs"]]

      pred_name = f"{self.data_output}/pred_{data_split}.parquet"
      if os.path.isfile(pred_name): os.system(f"rm {pred_name}")

      functions_to_apply = []
      functions_to_apply += ["untransform"]
      functions_to_apply += [
        partial(
          apply_classifier, 
          func=network.Predict, 
          X_columns=parameters['classifier'][self.parameter]["X_columns"],
        ),
        partial(self._WriteDataset,file_name=f"pred_{data_split}.parquet")
      ]
      pred_df.GetFull(
        method = None,
        functions_to_apply = functions_to_apply,
      )

      # Get ROC AUC from predictions
      if self.get_auc:

        if self.verbose:
          print("- Getting ROC AUC")

        pred_df = DataProcessor(
          [f"{self.data_output}/pred_{data_split}.parquet", f"{parameters['classifier'][self.parameter]['file_loc']}/y_{data_split}.parquet", f"{parameters['classifier'][self.parameter]['file_loc']}/wt_{data_split}.parquet"],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : parameters['classifier'][self.parameter],
          }
        )
        df = pred_df.GetFull(method="dataset")
        df = df[(df["wt"] > 0)]
        auc = roc_auc_score(df["classifier_truth"], df["probs"], sample_weight=df["wt"])
        performance_metrics[f"roc_auc_{data_split}"] = float(auc)
        print(f"ROC AUC for {data_split} dataset: {auc:.4f}")

    if self.get_auc:
      with open(f"{self.data_output}/performance_metrics.yaml", 'w') as yaml_file:
        yaml.dump(performance_metrics, yaml_file)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Add output data
    outputs += [f"{self.data_output}/pred_train.parquet"]
    if self.test_name is not None:
      outputs += [f"{self.data_output}/pred_{self.test_name}.parquet"]

    if self.get_auc:
      outputs += [f"{self.data_output}/performance_metrics.yaml"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initiate inputs
    inputs = []

    # Add parameters
    inputs += [self.parameters]

    # Add model
    inputs += [f"{self.model_input}/{self.model_name}/{self.file_name}_architecture.yaml"]
    inputs += [f"{self.model_input}/{self.model_name}/{self.file_name}.h5"]

    # Add data
    inputs += [
      f"{self.data_input}/X_train.parquet", 
      f"{self.data_input}/wt_train.parquet",
    ]
    if self.test_name is not None:
      inputs += [
        f"{self.data_input}/X_{self.test_name}.parquet",
        f"{self.data_input}/wt_{self.test_name}.parquet",
      ]

    return inputs

        