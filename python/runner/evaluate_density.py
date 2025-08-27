import os
import yaml

import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial

from data_processor import DataProcessor
from useful_functions import InitiateDensityModel, MakeDirectories

class EvaluateDensity():

  def __init__(self):
    """
    A class to create a new set of samples of the train and test datasets conditions to compare
    """
    # Default values - these will be set by the configure function
    self.parameters = None
    self.data_input = "data/"
    self.model_input = "models/"
    self.model_name = None
    self.file_name = None
    self.data_output = "data/"
    self.verbose = True     
    

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)


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


  def Run(self):
    """
    Run the code utilising the worker classes
    """

    # Open parameters file
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build the density model
    if self.verbose:
      print("- Building density network")
    density_model_name = f"{self.model_input}/{self.model_name}/{parameters['file_name']}"
    with open(f"{density_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
    network = InitiateDensityModel(
      architecture,
      self.data_input,
      options = {
        "data_parameters" : parameters["density"],
        "file_name" : self.file_name,
      }
    )

    # Loading density model
    if self.verbose:
      print(f"- Loading the density model {density_model_name}")
    network.Load(name=f"{density_model_name}.h5")

    def pred(df):
      df = network.Sample(df.loc[:,parameters["density"]["Y_columns"]],n_events=len(df))
      return df

    for tt in ["train", "test"]:

      if self.verbose:
        print(f"- Processing samples for the {tt} conditions")


      input_file = [f"{self.data_input}/{i}_{tt}.parquet" for i in ["X","Y"]]
      dp = DataProcessor(
        [input_file],
        "parquet",
        options = {
          "parameters" : parameters["density"]
        }
      )

      write_file_name = f"synth_{tt}.parquet"
      MakeDirectories(f"{self.data_output}/{write_file_name}")
      if os.path.isfile(f"{self.data_output}/{write_file_name}"):
        os.remove(f"{self.data_output}/{write_file_name}")    

      dp.GetFull(
        method=None,
        functions_to_apply=[
          "untransform",
          pred,
          "transform",
          partial(self._WriteDataset, file_name=write_file_name)
        ]
      )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    for tt in ["train", "test"]:
      outputs.append(f"{self.data_output}/synth_{tt}.parquet")

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    # Add data
    for tt in ["train", "test"]:
      inputs.append(f"{self.data_input}/Y_{tt}.parquet")

    # Add models
    for tt in ["train", "test"]:
      inputs.append(f"{self.model_input}/{self.model_name}/{self.file_name}.h5")
      inputs.append(f"{self.model_input}/{self.model_name}/{self.file_name}_architecture.yaml")

    # Add parameters
    inputs.append(self.parameters)

    return inputs

        