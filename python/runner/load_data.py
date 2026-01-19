import os
import re
import uproot

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial

from data_loader import DataLoader
from data_processor import DataProcessor
from useful_functions import GetCategoryLoop, GetParametersInModel, MakeDirectories, LoadConfig

pd.options.mode.chained_assignment = None

class LoadData():

  def __init__(self):

    # Required input which is the location of a file
    self.cfg = None

    # Required input which can just be parsed
    self.file_name = None

    # Other
    self.verbose = True
    self.data_output = "data/"
    self.batch_size = None
    self.columns = []


  def _GetTokens(self, input):
    tokens = re.findall(r"[A-Za-z_]\w*", input)
    reserved = {"and", "or", "not", "cos", "sin", "sinh", "cosh", "tanh", "abs", "exp", "sqrt"}
    return [t for t in tokens if t not in reserved]


  def _FindColumns(self, file_name, cfg):

    # Set up calculated
    calculated = []
    existing = []
    if "pre_calculate" in cfg["files"][file_name].keys():
      for k, v in cfg["files"][file_name]["pre_calculate"].items():
        if isinstance(v, str):
          if k not in self._GetTokens(v) and k not in existing:
            calculated += [k]
          else:
            existing += [k]
        else:
          calculated += v.get("outputs", [])

    # Add fitted variables
    self.columns = []

    # Add save extra columns from validation
    for k, v in cfg["validation"]["files"].items():
      if file_name == v:
        if k in cfg["preprocess"]["save_extra_columns"]:
          self.columns += cfg["preprocess"]["save_extra_columns"][k]

    # Add parameters in model
    self.columns += cfg["files"][file_name]["parameters"]

    # Add add columns
    if "add_columns" in cfg["files"][file_name].keys():
      self.columns += list(cfg["files"][file_name]["add_columns"].keys())

    # Add parameters and save extra columns from models
    for actual_file_name, model_types in cfg["models"].items():
      for model_type, models in model_types.items():
        for model in models:

          if "file" not in model.keys(): continue
          file_matched = (file_name == model["file"])
          if not file_matched: continue

          categories = GetCategoryLoop(cfg)
          if "categories" in model.keys():
            categories = model["categories"]

          if model_type == "yields":
            parameters_in_model = []
            for cat in categories:
              parameters_in_model += GetParametersInModel(actual_file_name, cfg, category=cat)
            parameters_in_model = sorted(list(set(parameters_in_model)))
            calculated += [i for i in parameters_in_model if i not in existing]

          if "parameters" in model.keys():
            calculated += model["parameters"]
          elif "parameter" in model.keys():
            calculated += [model["parameter"]]

          if actual_file_name in cfg["preprocess"]["save_extra_columns"]:
            self.columns += cfg["preprocess"]["save_extra_columns"][actual_file_name]

    calculated = sorted(list(set(calculated)))

    # Get from variables
    self.columns += [i for i in cfg["variables"] if i not in calculated]

    # Get from weight
    weight = cfg["files"][file_name]["weight"]
    self.columns += self._GetTokens(weight)

    # Get from post selection
    if "pre_calculate" in cfg["files"][file_name].keys():
      for k, v in cfg["files"][file_name]["pre_calculate"].items():
        if isinstance(v, str):
          self.columns += [i for i in self._GetTokens(v) if i not in calculated]
        else:
          inputs = v.get("inputs", [])
          self.columns += [i for i in inputs if i not in calculated]

    # Do post_calculate_selection
    if "post_calculate_selection" in cfg["files"][file_name].keys():
      if cfg["files"][file_name]["post_calculate_selection"] is not None:
        self.columns += [i for i in self._GetTokens(cfg["files"][file_name]["post_calculate_selection"]) if i not in calculated]

    # Get from weight shifts
    if "weight_shifts" in cfg["files"][file_name].keys():
      for k, v in cfg["files"][file_name]["weight_shifts"].items():
        if isinstance(v, str):
          self.columns += [i for i in self._GetTokens(v) if i not in calculated]
        else:
          inputs = v.get("inputs", [])
          self.columns += [i for i in inputs if i not in calculated]

    # Add categories
    if "categories" in cfg.keys():
      for k, v in cfg["categories"].items():
        self.columns += [i for i in self._GetTokens(v) if i not in calculated]

    self.columns = sorted(list(set(self.columns)))  # Remove duplicates


  def _DoWriteDataset(self, df, file_name):

    df = df.loc[:,self.columns]

    file_path = f"{self.data_output}/{file_name}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(file_path):
      combined_table = pa.concat_tables([pq.read_table(file_path), table])
      pq.write_table(combined_table, file_path, compression='snappy')
    else:
      pq.write_table(table, file_path, compression='snappy')

    return df

  def _MakeFiles(self, file_name, file_info):

    if self.verbose:
      print(f"- Making {file_name}")

    for input_file_ind, input_file in enumerate(file_info["inputs"]):

      if self.verbose:
        print(f"- Loading file {input_file}")

      batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_PREPROCESS")) if self.batch_size is None else self.batch_size

      if ".root" in input_file:
        # Initiate root file
        file = uproot.open(input_file)
        loader = file[file_info["tree_name"]]
        total_entries = loader.num_entries
      elif ".parquet" in input_file:
        # Initiate data loader
        loader = DataLoader(input_file,batch_size=batch_size)
        total_entries = loader.num_rows
      else:
        raise ValueError("Input files can only be root or parquet files")

      # Loop through batches
      for start in range(0, total_entries, batch_size):
        stop = min(start + batch_size, total_entries)

        if ".root" in input_file:
          arrays = loader.arrays(
            None,
            cut=file_info["selection"],
            entry_start=start,
            entry_stop=stop 
          )   
          df = pd.DataFrame(np.array(arrays))
        elif ".parquet" in input_file:
          df = loader.LoadNextBatch()
          if file_info.get("selection", False):
            df = df.loc[df.eval(file_info["selection"]),:]


        # Skip if dataframe is empty
        if len(df) == 0: continue

        # Add extra columns
        if "add_columns" in file_info.keys():
          new_cols = {}
          for extra_col_name, extra_col_value in file_info["add_columns"].items():
            if isinstance(extra_col_value, list):
              df = df.assign(**{extra_col_name: extra_col_value[input_file_ind]})
            elif isinstance(extra_col_value, (float, int)):
              df = df.assign(**{extra_col_name: extra_col_value})
            else:
              raise ValueError(f"Unknown type for extra column {extra_col_name}: {type(extra_col_value)}")

        # Post add column selection
        if "post_add_column_selection" in file_info.keys():
          if file_info["post_add_column_selection"] is not None:
            df = df.query(file_info["post_add_column_selection"])

        # Selection per file
        if "per_file_selection" in file_info.keys():
          if file_info["per_file_selection"] is not None:
            df = df.query(file_info["per_file_selection"][input_file_ind])

        # Keep only required columns
        df = df.loc[:, self.columns]
    
        # Removing nans
        nan_rows = df[df.isna().any(axis=1)]
        if len(nan_rows) > 0:
          if self.verbose:
            print(f"Removing {len(nan_rows)}/{len(df)} rows with NaNs")
            print(nan_rows.head())
          df = df.dropna()

        # Set type
        df = df.astype(np.float64)

        ## Remove negative weights
        #if "remove_negative_weights" in file_info.keys():
        #  if file_info["remove_negative_weights"]:
        #    neg_weight_rows = (df.loc[:,"wt"] < 0)
        #    if len(df[neg_weight_rows]) > 0:
        #      if self.verbose: 
        #        print(f"Total negative weights: {len(df[neg_weight_rows])}/{len(df)} = {round(len(df[neg_weight_rows])/len(df),4)}")
        #      df = df[~neg_weight_rows]

        # Write dataset
        self._DoWriteDataset(df, file_name)

    # Add summed column
    if "add_summed_columns" in file_info.keys():
      if self.verbose:
        print("- Adding summed variables")
      
      filein = f"{file_name}"
      fileout = f"{file_name}_add_summed_columns"

      # Get sums
      sums = []
      dp = DataProcessor(
        [f"{self.data_output}/{filein}.parquet"],
        "parquet",
        options = {},
        batch_size=self.batch_size,
      )
      for summed_col in file_info["add_summed_columns"]:
        if summed_col["column"] is not None:
          sums.append(
            dp.GetFull(
              method="sum_formula",
              column=summed_col["column"],
              extra_sel=summed_col["selection"]
            )
          )
        else:
          sums.append(
            dp.GetFull(
              method="count",
              extra_sel=summed_col["selection"]
            )
          )          

      # Apply sums
      if os.path.isfile(f"{self.data_output}/{fileout}.parquet"):
        os.system(f"rm {self.data_output}/{fileout}.parquet")

      def apply_sum(df, sums, summed_col_info):
        unique_names = []
        for summed_col in summed_col_info:
          unique_names = sorted(list(set(unique_names+[summed_col["name"]])))
        for col in unique_names:
          df.loc[:, col] = 0.0
        for ind, summed_col in enumerate(summed_col_info):
          if summed_col["selection"] is not None:
            df.loc[df.eval(summed_col["selection"]),summed_col["name"]] = sums[ind]
          else:
            df.loc[:,summed_col["name"]] = sums[ind]
        return df
      
      dp.GetFull(
        method=None,
        functions_to_apply=[
          partial(apply_sum, sums=sums, summed_col_info=file_info["add_summed_columns"]),
          partial(self._DoWriteDataset, file_name=fileout)
        ]
      )
      os.system(f"mv {self.data_output}/{fileout}.parquet {self.data_output}/{filein}.parquet")



  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)


  def Run(self):

    # Load config
    cfg = LoadConfig(self.cfg)

    self._FindColumns(self.file_name, cfg)

    # Make file
    MakeDirectories(f"{self.data_output}/{self.file_name}.parquet")
    if os.path.isfile(f"{self.data_output}/{self.file_name}.parquet"):
      os.system(f"rm {self.data_output}/{self.file_name}.parquet")
    self._MakeFiles(self.file_name, cfg["files"][self.file_name])


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [f"{self.data_output}/{self.file_name}.parquet"]
    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [self.cfg]
    return inputs
