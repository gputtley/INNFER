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
    self.selection = None
    self.extra_selection = None
    self.data_input = None
    self.data_output = None
    self.verbose = True
    self.add_columns = {}

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if isinstance(self.data_input,str):
      self.data_input = [self.data_input]

  def _AddColumns(self, df, ind):
    for col, value in self.add_columns.items():
      if isinstance(value, list):
        df[col] = value[ind]
      else:
        df[col] = value
    return df

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    if os.path.isfile(f"{self.data_output}/data.parquet"):
      os.system(f"rm {self.data_output}/data.parquet")

    if self.selection is None and self.extra_selection is None:
      selection = None
    elif self.selection is not None and self.extra_selection is None:
      selection = self.selection
    elif self.selection is None and self.extra_selection is not None:
      selection = self.extra_selection
    else:
      selection = f"({self.selection}) & ({self.extra_selection})"

    for ind, k in enumerate(self.data_input):

      if ".parquet" in k:
        file_type = "parquet"
      elif ".root" in k:
        file_type = "root"
      else:
        raise ValueError("Unsupported file type. Please provide a .parquet or .root file.")

      dp = DataProcessor(
          k,
          file_type,
          options = {
            "selection" : selection,
          }
        )

      dp.GetFull(
        method=None,
        functions_to_apply=[
          partial(self._AddColumns, ind=ind),
          "selection",
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
    inputs = self.data_input
    return inputs

        