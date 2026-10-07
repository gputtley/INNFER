import os
import yaml

import numpy as np
import pandas as pd

from data_processor import DataProcessor


class DensityCorrection():
  """
  Evaluate the simulation-to-flow classifier ratio on physical samples.
  """

  def __init__(self):

    self.model_input = None
    self.parameters = None
    self.density_model = None
    self.density_parameters = None


  def Configure(self, options):
    """
    Configure the correction checkpoint and its saved training transforms.
    """
    for key, value in options.items():
      setattr(self, key, value)


  def Load(self):
    """
    Load the classifier without requiring its original training datasets.
    """
    from fcnn_network import FCNNNetwork

    with open(self.parameters, 'r') as file:
      metadata = yaml.safe_load(file)
    with open(f"{self.model_input}_architecture.yaml", 'r') as file:
      architecture = yaml.safe_load(file)
    if architecture["type"] != "FCNN" or metadata["classes"] != {0: "synthetic", 1: "simulation"}:
      raise ValueError("Density correction requires a simulation-to-flow FCNN classifier")
    if os.path.normpath(metadata["density_model"]) != os.path.normpath(self.density_model):
      raise ValueError("Density correction was trained for a different density model")
    saved_parameters = {k: v for k, v in metadata["density_parameters"].items() if k != "file_loc"}
    current_parameters = {k: v for k, v in self.density_parameters.items() if k != "file_loc"}
    if saved_parameters != current_parameters:
      raise ValueError("Density correction transforms do not match the density parameters")

    self.data_parameters = metadata["density_parameters"]
    self.feature_columns = metadata["classifier"]["X_columns"]
    self.network = FCNNNetwork(options={
      **{k: v for k, v in architecture.items() if k != "type"},
      "data_parameters": metadata["classifier"],
    })
    self.network.Load(name=f"{self.model_input}.h5")
    self.transformer = DataProcessor(
      [[pd.DataFrame()]], "dataset",
      options={"parameters": self.data_parameters},
    )


  def Predict(self, df, conditions):
    """
    Return p_simulation(X|Y)/p_flow(X|Y), using the training feature space.
    """
    features = df.loc[:, self.data_parameters["X_columns"]].copy()
    for column in self.data_parameters["Y_columns"]:
      features[column] = conditions[column]
    features = self.transformer.TransformData(features)
    features = features.loc[:, self.feature_columns]
    if not np.all(np.isfinite(features.to_numpy())):
      raise ValueError("Non-finite density correction features after transformation")
    probabilities = np.asarray(self.network.Predict(features, transform_X=False), dtype=np.float64)
    if probabilities.shape != (len(df), 2) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities[:, 0] <= 0):
      raise ValueError("Density correction requires finite probabilities and a positive synthetic denominator")
    ratio = probabilities[:, 1] / probabilities[:, 0]
    if not np.all(np.isfinite(ratio)):
      raise ValueError("Non-finite density correction ratio")
    return ratio


  def SamplingSupport(self, df):
    """
    Identify events inside the same observable ranges used by flow sampling.
    """
    support = np.ones(len(df), dtype=bool)
    transformed = self.transformer.TransformData(df.loc[:, self.data_parameters["X_columns"]].copy()) if "minmax" in self.data_parameters else df
    for key, data in [("minmax", transformed), ("initial_minmax", df)]:
      for column, bounds in self.data_parameters.get(key, {}).items():
        if column in data.columns and column in self.data_parameters["X_columns"]:
          support &= (data[column].to_numpy() > bounds["min"]) & (data[column].to_numpy() < bounds["max"])
    return support


  def Inputs(self):
    """
    Declare the trained classifier and the metadata defining its transforms.
    """
    if self.model_input is None or self.parameters is None:
      raise ValueError("Density correction requires a trained classifier and its parameters.yaml")
    inputs = [self.parameters, f"{self.model_input}_architecture.yaml", f"{self.model_input}.h5"]
    if os.path.isfile(self.parameters):
      with open(self.parameters, 'r') as file:
        parameters = yaml.safe_load(file)["density_parameters"]
      inputs += list(parameters.get("spline_to_gaussian", {}).get("forward", {}).values())
      if "pca_whitening" in parameters:
        inputs += [parameters["pca_whitening"]["location"]]
    return list(dict.fromkeys(inputs))
