import os

import pyarrow as pa
import pyarrow.parquet as pq

from data_processor import DataProcessor
from useful_functions import MakeDirectories
from write_parquet import WriteParquet

class ResampleValidationForData():

  def __init__(self):
    # Default values - these will be set by the configure function
    self.data_output = None
    self.data_inputs = None
    self.sim_type = "val"
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

    wp = WriteParquet(
      name = self.data_output.split("/")[-1].replace(".parquet", ""),
      data_output = "/".join(self.data_output.split("/")[:-1]),
    )

    for file_name, file_loc in self.data_inputs.items():

      if self.verbose:
        print(f"Processing file: {file_name}")
      
      files = [f"{file_loc}/X_{self.sim_type}.parquet", f"{file_loc}/wt_{self.sim_type}.parquet"]

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
          wp
        ]
      )

    wp.collect()

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
    for _, file_loc in self.data_inputs.items():
      inputs += [f"{file_loc}/X_{self.sim_type}.parquet", f"{file_loc}/wt_{self.sim_type}.parquet"]
    return inputs

        