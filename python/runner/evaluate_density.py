import yaml

import numpy as np
import pandas as pd

from data_processor import DataProcessor
from useful_functions import InitiateDensityModel
from write_parquet import WriteParquet

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
    self.train_name = "train"
    self.test_name = "test"
    self.seed = 42
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
      train_name = self.train_name,
      options = {
        "data_parameters" : parameters["density"],
        "file_name" : self.file_name,
      }
    )

    # Loading density model
    if self.verbose:
      print(f"- Loading the density model {density_model_name}")
    network.Load(name=f"{density_model_name}.h5")
    network.use_gaussian_cache = False

    for split_index, (source_split, tt) in enumerate([(self.train_name, "train"), (self.test_name, "test")]):

      if self.verbose:
        print(f"- Processing samples for the {tt} conditions")


      input_file = [f"{self.data_input}/X_{source_split}.parquet"]
      if parameters["density"]["Y_columns"]:
        input_file += [f"{self.data_input}/Y_{source_split}.parquet"]
      dp = DataProcessor(
        [input_file],
        "parquet",
        options = {
          "parameters" : parameters["density"]
        }
      )

      if any(loader.num_rows != dp.data_loaders[0][0].num_rows for loader in dp.data_loaders[0]):
        raise ValueError(f"Misaligned density input tables for {source_split}")
      batch_index = 0

      def pred(df):
        nonlocal batch_index
        Y = df.loc[:, parameters["density"]["Y_columns"]].reset_index(drop=True)
        if architecture["type"] == "BayesFlow":
          # Both the source conditions and saved samples are in training space.
          # Avoid an inverse/forward transform round trip and range filtering,
          # which can otherwise remove the final one-row batch.
          synth = network.Sample(
            Y, n_events=len(df), seed=self.seed + split_index,
            batch_number=batch_index, batch_size=dp.batch_size,
            transform_Y=False, transform_X=False,
          )
        else:
          physical_Y = DataProcessor([[Y]], "dataset", options={"parameters": parameters["density"]}).GetFull(method="dataset", functions_to_apply=["untransform"]) if len(Y.columns) else Y
          synth = network.Sample(physical_Y, n_events=len(df))
          synth = DataProcessor([[synth]], "dataset", options={"parameters": parameters["density"]}).GetFull(method="dataset", functions_to_apply=["transform"])
        batch_index += 1
        if synth is not None:
          synth = synth.loc[:, parameters["density"]["X_columns"]]
        if synth is None or len(synth) != len(df) or not np.all(np.isfinite(synth.to_numpy())):
          raise ValueError("EvaluateDensity must generate one finite row per source condition")
        # Save the conditions as well, so downstream pairing can verify order.
        return pd.concat([synth.reset_index(drop=True), Y], axis=1)
  
      wp = WriteParquet(
        name = f"synth_{tt}",
        data_output = self.data_output,
      )
      dp.GetFull(
        method=None,
        functions_to_apply=[
          pred,
          wp
        ]
      )
      wp.collect(memory_safe=True)


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
    for tt in [self.train_name, self.test_name]:
      inputs += [f"{self.data_input}/{key}_{tt}.parquet" for key in ["X", "Y"]]

    # Add models
    inputs.append(f"{self.model_input}/{self.model_name}/{self.file_name}.h5")
    inputs.append(f"{self.model_input}/{self.model_name}/{self.file_name}_architecture.yaml")

    # Add parameters
    inputs.append(self.parameters)

    return inputs

