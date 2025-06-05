import os

import pyarrow as pa
import pyarrow.parquet as pq

from data_processor import DataProcessor
from useful_functions import MakeDirectories

class ResampleValidationForData():

  def __init__(self):
    # Default values - these will be set by the configure function
    self.data_output = None
    self.data_inputs = None
    self.verbose = True

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

    if self.data_output is None:
      raise ValueError("data_output must be set before running.")

    # Check if data already exists
    if os.path.exists(self.data_output):
      print(f"Output directory {self.data_output} already exists. Please remove it before running.")
      return

    for file_name, v1 in self.data_inputs.items():
      for category, v2 in v1.items():

        if self.verbose:
          print(f"Processing file: {file_name}, category: {category}")
        
        files = [f"{v2}/X_val.parquet", f"{v2}/wt_val.parquet"]
        extra_file = f"{v2}/Extra_val.parquet"
        if os.path.isfile(extra_file):
          files.append(extra_file)

        dp = DataProcessor(
            files,
            "parquet",
            wt_name = "wt",
            options = {
              "resample" : True,
              "resample_drop_negative_weights" : True,
            },
          )

        dp.GetFull(
          method=None,
          functions_to_apply=[
            self._WriteDataset
          ]
        )


  def _WriteDataset(self, df):

    df = df.drop(columns=["wt"])
    MakeDirectories(self.data_output)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(self.data_output):
      existing_table = pq.read_table(self.data_output)
      common_columns = set(existing_table.column_names).intersection(set(table.column_names))
      table = table.select(common_columns)
      existing_table = existing_table.select(common_columns)
      combined_table = pa.concat_tables([existing_table, table])
      pq.write_table(combined_table, self.data_output, compression='snappy')
    else:
      pq.write_table(table, self.data_output, compression='snappy')

    return df

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [self.data_output]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for _, v1 in self.data_inputs.items():
      for _, v2 in v1.items():        
        inputs += [f"{v2}/X_val.parquet", f"{v2}/wt_val.parquet"]
    return inputs

        