import os

import pyarrow as pa
import pyarrow.parquet as pq

from useful_functions import MakeDirectories

class WriteParquet:

  def __init__(self, name="file", data_output="data", rename_columns={}, rename_file_column={}):
    """
    Class to write a dataframe to parquet files, it can write to multiple files with different columns and rename columns before writing

    parameters:
    name: str or dict, if str, the name of the file to write, if dict, the keys are the names of the files to write and the values are the columns to write to each file
    data_output: str, the directory to write the files to
    rename_columns: dict, the keys are the old column names and the values are the new column names to rename in the dataframe before writing
    rename_file_column: dict, the keys are the file names and the values are the column names to rename in the dataframe before writing to that file
    """
    if isinstance(name, dict):
      self.name = name
    elif isinstance(name, str):
      self.name = {name: None}
    else:
      raise ValueError("name must be a string of the name or a dictionary of the names and columns to write")
    self.data_output = data_output
    self.rename_columns = rename_columns
    self.rename_file_column = rename_file_column
    self.index = 1
    self.created_files = {}

  def __call__(self, df):
    """
    Write the dataframe to parquet files, it can write to multiple files with different columns and rename columns before writing
    parameters:
    df: pandas dataframe, the dataframe to write to parquet files
    """
    if df is None:
      return None

    for old_name, new_name in self.rename_columns.items():
      df = df.rename(columns={old_name: new_name})

    for file_name, column_name in self.rename_file_column.items():
      if file_name in self.name:
        if len(self.name[file_name]) == 1:
          df = df.rename(columns={self.name[file_name][0]: column_name})

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

  def collect(self, verbose=False):
    """
    Collect the parquet files created by the __call__ method and combine them into a single parquet file for each key in the name dictionary, then remove the individual files
    """
    for k in self.name.keys():

      # Remove existing file if exists
      file_name = f"{self.data_output}/{k}.parquet"
      if os.path.exists(file_name):
        if verbose:
          print(f"Removing existing file {file_name}")
        os.remove(file_name)

      # Skip if no files created for this key
      if k not in self.created_files:
        if verbose:
          print(f"No files created for {k}, skipping")
        continue

      # Combine tables
      if verbose:
        print(f"Combining {len(self.created_files[k])} files for {k}")
      tables = [pq.read_table(f) for f in self.created_files[k]]
      combined_table = pa.concat_tables(tables)

      # Write combined table to file
      MakeDirectories(file_name)
      pq.write_table(combined_table, file_name, compression='snappy')
      if verbose:
        print(f"Written combined file {file_name}")
      for file in self.created_files[k]:
        if verbose:          
          print(f"Removing file {file}")
        os.remove(file)
