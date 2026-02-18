import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from useful_functions import MakeDirectories


class WriteParquet:

  def __init__(self, name="file", data_output="data"):
    if isinstance(name, dict):
      self.name = name
    elif isinstance(name, str):
      self.name = {name: None}
    else:
      raise ValueError("name must be a string of the name or a dictionary of the names and columns to write")
    self.data_output = data_output
    self.index = 1
    self.created_files = {}

  def __call__(self, df):
    for k, v in self.name.items():
      copy_df = df.copy()
      if v is not None:
        copy_df = copy_df[v]
      file_path = f"{self.data_output}/{k}_{self.index}.parquet"
      if os.path.exists(file_path):
        os.remove(file_path)
      MakeDirectories(file_path)
      table = pa.Table.from_pandas(copy_df, preserve_index=False)
      pq.write_table(table, file_path, compression='snappy')
      if k not in self.created_files:
        self.created_files[k] = []
      self.created_files[k].append(file_path)
    self.index += 1
    return df

  def collect(self):
    for k in self.name.keys():
      tables = [pq.read_table(f) for f in self.created_files[k]]
      combined_table = pa.concat_tables(tables)
      file_name = f"{self.data_output}/{k}.parquet"
      if os.path.exists(file_name):
        os.remove(file_name)
      MakeDirectories(file_name)
      pq.write_table(combined_table, file_name, compression='snappy')
      for file in self.created_files[k]:
        os.remove(file)
