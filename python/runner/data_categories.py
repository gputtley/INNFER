import os

import pyarrow as pa
import pyarrow.parquet as pq

from data_processor import DataProcessor
from functools import partial
from useful_functions import MakeDirectories

class DataCategories():

  def __init__(self):
    """
    A template class.
    """
    self.extra_selection = None
    self.data_input = None
    self.data_output = None
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

    if ".parquet" in self.data_input:
      file_type = "parquet"
    elif ".root" in self.data_input:
      file_type = "root"
    else:
      raise ValueError("Unsupported file type. Please provide a .parquet or .root file.")

    dp = DataProcessor(
        self.data_input,
        file_type,
        options = {
          "selection" : self.extra_selection,
        }
      )

    if os.path.isfile(f"{self.data_output}/data.parquet"):
      os.system(f"rm {self.data_output}/data.parquet")

    dp.GetFull(
      method=None,
      functions_to_apply=[
        partial(self._WriteDataset, file_name="data.parquet"),
      ]
    )

    if self.verbose:
      print(f"Created file {self.data_output}/data.parquet")


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
  

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [f"{self.data_output}/data.parquet",]
    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [self.data_input]
    return inputs

        