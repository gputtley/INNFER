import yaml

import numpy as np

from multidim_metrics import MultiDimMetrics
from useful_functions import MakeDirectories

class C2ST():

  def __init__(self):
    """
    A template class.
    """
    self.model = None
    self.architecture = None
    self.parameters = None
    self.data_output = "data/"
    self.seed = 0
    self.verbose = True
    self.sim_type = "test"

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
    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      self.open_parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the architecture in
    if self.verbose:
      print("- Loading in the architecture")
    with open(self.architecture, 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build model
    if self.verbose:
      print("- Building the model")
    from network import Network
    self.network = Network(
      f"{self.open_parameters['file_loc']}/X_train.parquet",
      f"{self.open_parameters['file_loc']}/Y_train.parquet", 
      f"{self.open_parameters['file_loc']}/wt_train.parquet", 
      f"{self.open_parameters['file_loc']}/X_{self.sim_type}.parquet",
      f"{self.open_parameters['file_loc']}/Y_{self.sim_type}.parquet", 
      f"{self.open_parameters['file_loc']}/wt_{self.sim_type}.parquet",
      options = {
        **architecture,
        **{
          "data_parameters" : self.open_parameters
        }
      }
    )  
    
    # Loading model
    if self.verbose:
      print("- Loading the model")
    self.network.Load(name=self.model)

    # Make multidimensional metrics
    if self.verbose:
      print("- Getting C2ST metrics")

    mm = MultiDimMetrics(
      self.network,
      self.open_parameters,
      {self.sim_type:[f"{self.open_parameters['file_loc']}/X_{self.sim_type}.parquet", f"{self.open_parameters['file_loc']}/Y_{self.sim_type}.parquet", f"{self.open_parameters['file_loc']}/wt_{self.sim_type}.parquet"]}
    )
    mm.verbose = self.verbose
    mm.resample_sim = True
    mm.AddBDTSeparation()
    res = mm.Run(seed=self.seed)

    out = {}
    for k, v in res.items():
      if "auc" in k:
        out["auc"] = {self.seed: float(v)}
      elif "accuracy" in k:
        out["accuracy"] = {self.seed: float(v)}

    # Write out the results to yaml
    if self.verbose:
      print("- Writing the results to yaml")
    MakeDirectories(self.data_output)
    with open(f"{self.data_output}/c2st_{self.sim_type}_{self.seed}.yaml", 'w') as file:
      yaml.dump(out, file)



  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [f"{self.data_output}/c2st_{self.sim_type}_{self.seed}.yaml"]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    inputs = [
      self.model,
      self.architecture,
      self.parameters,
      f"{parameters['file_loc']}/X_train.parquet",
      f"{parameters['file_loc']}/Y_train.parquet", 
      f"{parameters['file_loc']}/wt_train.parquet", 
      f"{parameters['file_loc']}/X_{self.sim_type}.parquet",
      f"{parameters['file_loc']}/Y_{self.sim_type}.parquet", 
      f"{parameters['file_loc']}/wt_{self.sim_type}.parquet",
    ]
    return inputs

        