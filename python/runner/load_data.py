import os
import uproot

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial

from data_loader import DataLoader
from data_processor import DataProcessor
from useful_functions import MakeDirectories, LoadConfig

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
    self.batch_size = 10**7

  def _DoWriteDataset(self, df, file_name):

    file_path = f"{self.data_output}/{file_name}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(file_path):
      existing_table = pq.read_table(file_path)
      common_columns = list(set(existing_table.column_names) & set(table.column_names))
      table = table.select(common_columns)
      existing_table = existing_table.select(common_columns)
      combined_table = pa.concat_tables([existing_table, table])
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

      batch_size = int(os.getenv("EVENTS_PER_BATCH")) if self.batch_size is None else self.batch_size

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
          for extra_col_name, extra_col_value in file_info["add_columns"].items():
            if isinstance(extra_col_value, list):
              df.loc[:,extra_col_name] = extra_col_value[input_file_ind]
            elif isinstance(extra_col_value, float) or isinstance(extra_col_value, int):
              df.loc[:,extra_col_name] = extra_col_value
            else:
              raise ValueError(f"Unknown type for extra column {extra_col_name}: {type(extra_col_value)}")

        """
        # Calculate weight
        if "weight" in file_info.keys():
          df.loc[:,"wt"] = df.eval(file_info["weight"])
          df = df.loc[(df.loc[:,"wt"]!=0),:]
        else:
          df.loc[:,"wt"] = 1.0

        # Scale 
        if "scale" in file_info.keys():
          if isinstance(file_info["scale"], int) or isinstance(file_info["scale"], float):
            df.loc[:,"wt"] *= file_info["scale"]
          else:
            df.loc[:,"wt"] *= file_info["scale"][input_file_ind]
        """
            
        # Removing nans
        nan_rows = df[df.isna().any(axis=1)]
        if len(nan_rows) > 0:
          df = df.dropna()

        # Set type
        df = df.astype(np.float64)

        # Remove negative weights
        if "remove_negative_weights" in file_info.keys():
          if file_info["remove_negative_weights"]:
            neg_weight_rows = (df.loc[:,"wt"] < 0)
            if len(df[neg_weight_rows]) > 0:
              if self.verbose: 
                print(f"Total negative weights: {len(df[neg_weight_rows])}/{len(df)} = {round(len(df[neg_weight_rows])/len(df),4)}")
              df = df[~neg_weight_rows]

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
          unique_names = list(set(unique_names+[summed_col["name"]]))
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
