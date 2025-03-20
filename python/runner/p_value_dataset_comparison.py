import copy
import yaml

import numpy as np

from data_processor import DataProcessor
from multidim_metrics import MultiDimMetrics
from useful_functions import MakeDirectories

class PValueDatasetComparison():

  def __init__(self):
    """
    A template class.
    """
    self.comparison_type = "sim_vs_synth"
    self.model = None
    self.architecture = None
    self.parameters = None
    self.data_output = "data/"
    self.seed = 0
    self.verbose = True
    self.sim_type = "val"

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

    sim_files = [f"{self.open_parameters['file_loc']}/X_{self.sim_type}.parquet", f"{self.open_parameters['file_loc']}/Y_{self.sim_type}.parquet", f"{self.open_parameters['file_loc']}/wt_{self.sim_type}.parquet"]

    mm = MultiDimMetrics(
      self.network,
      self.open_parameters,
      {self.sim_type:sim_files}
    )
    mm.verbose = self.verbose
    mm.AddBDTSeparation()
    mm.AddWassersteinSliced()
    mm.AddWassersteinUnbinned()
    mm.AddKMeansChiSquared()

    if self.verbose:
      print(f" - Building datasets")

    # Get simulated data
    sim_dp = DataProcessor(
      sim_files,
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : self.open_parameters
      }
    )
    mm.sim_dataset = sim_dp.GetFull("dataset", functions_to_apply=["untransform"])    

    import tensorflow as tf
    tf.random.set_seed(self.seed)
    tf.keras.utils.set_random_seed(self.seed)    

    if self.comparison_type == "SimVsSynth":
      extra_name = ""
      mm.MakeDatasets(None,only_synth=True,synth_size="n_eff_conditions")
    elif self.comparison_type == "SynthVsSynth":  
      extra_name = f"_{self.seed}"
      mm.MakeDatasets(None,only_synth=True,synth_size="n_eff_conditions")
      syth_dataset_seed_1 = copy.deepcopy(mm.synth_dataset)
      tf.random.set_seed(self.seed+1)
      tf.keras.utils.set_random_seed(self.seed+1)    
      mm.MakeDatasets(None,only_synth=True,synth_size="n_eff_conditions")      
      mm.sim_dataset = syth_dataset_seed_1

    if self.verbose:
      print(f" - Calculating metrics")

    out = mm.Run(make_datasets=False)

    # Write out the results to yaml
    if self.verbose:
      print("- Writing the results to yaml")
    MakeDirectories(self.data_output)
    with open(f"{self.data_output}/p_value_dataset_comparison_{self.sim_type}{extra_name}.yaml", 'w') as file:
      yaml.dump(out, file)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
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

        