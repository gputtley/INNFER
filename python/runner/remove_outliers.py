import copy
import os
import yaml

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial

from data_processor import DataProcessor

class RemoveOutliers():

  def __init__(self):

    self.cfg = None
    self.verbose = True
    self.data_splits = ["train","test","test_inf","val","full"]
    self.batch_size = 10**7

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def _DoWriteAllFromDataProcesor(self, df, X_columns, Y_columns, X_file_path, Y_file_path, wt_file_path, extra_file_path="", extra_columns=[]):
    file_path_translate = {"X":X_file_path, "Y":Y_file_path, "wt":wt_file_path}
    loop_over = {"X":X_columns, "Y":Y_columns, "wt":["wt"]}
    if len(extra_columns) > 0:
      loop_over["Extra"] = [col for col in extra_columns if col in df.columns]
      file_path_translate["Extra"] = extra_file_path
    for data_type, columns in loop_over.items():
      file_path = file_path_translate[data_type]
      table = pa.Table.from_pandas(df.loc[:,columns], preserve_index=False)
      if os.path.isfile(file_path):
        combined_table = pa.concat_tables([pq.read_table(file_path), table])
        pq.write_table(combined_table, file_path, compression='snappy')
      else:
        pq.write_table(table, file_path, compression='snappy')
    return df

  def _get_parameters(self,cfg):
    parameters = {}
    for file_name in cfg["files"].keys():
      parameter_file_name = f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml"
      with open(parameter_file_name, 'r') as yaml_file:
        parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return parameters

  def Run(self):

    # Open config
    with open(self.cfg, 'r') as yaml_file:
      cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load parameter files
    parameters = self._get_parameters(cfg)


    training_ranges = {}
    for file_name, params in parameters.items():

      # Get training files
      file_loop = ["X","Y","wt"]
      if "Extra_columns" in params.keys():
        if len(params["Extra_columns"]) > 0:
          file_loop += ["Extra"]
      file_names = [f'{params["file_loc"]}/{i}_train.parquet' for i in file_loop]

      # Build data loader
      dp = DataProcessor(
        [file_names],
        "parquet",
        options = {
          "parameters" : params,
          "functions" : ["untransform"]
        }
      )

      # Get min max
      min_max = dp.GetFull(method="min_max")
    
      if training_ranges == {}:
        training_ranges = {k:v for k, v in min_max.items() if k in params["X_columns"]}
      else:
        for k in params["X_columns"]:
          training_ranges[k] = [max(training_ranges[k][0],min_max[k][0]),min(training_ranges[k][1],min_max[k][1])]

    if self.verbose:
      print(f"Capping ranges to: {training_ranges}")


    # Apply selection to all files
    for file_name, params in parameters.items():
      for data_split in self.data_splits:

        # Get files
        file_loop = ["X","Y","wt"]
        if "Extra_columns" in params.keys():
          if len(params["Extra_columns"]) > 0:
            file_loop += ["Extra"]
        file_names = [f'{params["file_loc"]}/{i}_{data_split}.parquet' for i in file_loop]
        removed_outliers_names = [f'{params["file_loc"]}/{i}_{data_split}_removed_outliers.parquet' for i in file_loop]

        if data_split == "full":
          initial_function = []
          final_function = []
        else:
          initial_function = ["untransform"]
          final_function = ["transform"]

        min_selection = " & ".join([f"({k}>={v[0]})" for k,v in training_ranges.items()])
        max_selection = " & ".join([f"({k}<={v[1]})" for k,v in training_ranges.items()])
        selection = f"({min_selection}) & ({max_selection})"

        dp = DataProcessor(
          [file_names],
          "parquet",
          options = {
            "wt_name" : "wt",
            "parameters" : params,
            "selection" : selection
          },
          batch_size=self.batch_size,
        )

        for removed_outliers_name in removed_outliers_names:
          if os.path.isfile(removed_outliers_name):
            os.system(f"rm {removed_outliers_name}")

        dp.GetFull(
          method = None,
          functions_to_apply = initial_function + final_function + [partial(self._DoWriteAllFromDataProcesor, X_columns=params["X_columns"], Y_columns=params["Y_columns"], X_file_path=removed_outliers_names[0], Y_file_path=removed_outliers_names[1], wt_file_path=removed_outliers_names[2], extra_file_path=removed_outliers_names[3] if len(removed_outliers_names)>3 else "", extra_columns=params["Extra_columns"] if len(removed_outliers_names)>3 else [])],
        )

        for ind, removed_outliers_name in enumerate(removed_outliers_names):
          if os.path.isfile(removed_outliers_name):
            os.system(f"mv {removed_outliers_name} {file_names[ind]}")


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