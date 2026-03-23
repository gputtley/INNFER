import importlib
import os

import pyarrow as pa
import pyarrow.parquet as pq

from data_processor import DataProcessor
from functools import partial
from useful_functions import MakeDirectories
from write_parquet import WriteParquet

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
    self.calculate = {}

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
        #df[col] = value[ind]
        df = df.assign(**{col: value[ind]})
      else:
        #df[col] = value
        df = df.assign(**{col: value})
    return df


  def _Calculate(self, df):
    for name, value in self.calculate.items():
      if isinstance(value, str):
        df[name] = df.eval(value)
      elif isinstance(value, dict):
        if value["type"] == "function":
          module = importlib.import_module(value["file"])
          func = getattr(module, value["name"])
          df = func(df, **value["args"])
      else:
        raise ValueError(f"Calculate type {type(value)} not recognised")
    return df


  def Run(self):
    """
    Run the code utilising the worker classes
    """

    if self.selection is None and self.extra_selection is None:
      selection = None
    elif self.selection is not None and self.extra_selection is None:
      selection = self.selection
    elif self.selection is None and self.extra_selection is not None:
      selection = self.extra_selection
    else:
      selection = f"({self.selection}) & ({self.extra_selection})"

    wp = WriteParquet(
      name = "data",
      data_output = self.data_output,
    )

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
          self._Calculate,
          "selection",
          wp,
        ]
      )
    wp.collect()

    if self.verbose:
      print(f"Created file {self.data_output}/data.parquet")
  

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

        