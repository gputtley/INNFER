import copy
import os
import wandb
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from random_word import RandomWords
from scipy.interpolate import CubicSpline

from data_processor import DataProcessor
from useful_functions import InitiateClassifierModel,MakeDirectories

class TrainClassifier():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None
    self.architecture = None

    # other
    self.file_name = None
    self.data_input = None
    self.parameter = None
    self.use_wandb = False
    self.initiate_wandb = False
    self.wandb_project_name = "innfer"
    self.wandb_submit_name = "innfer"
    self.verbose = True
    self.disable_tqdm = False
    self.data_output = "data/"
    self.plots_output = "plots/"
    self.save_extra_name = ""
    self.test_name = "test"
    self.no_plot = False
    self.save_model_per_epoch = False
    self.model_type = "FCNN"


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
  

  def _NormaliseConditions(self):

    train_data = [f"{self.data_input}/X_train.parquet", f"{self.data_input}/y_train.parquet", f"{self.data_input}/wt_train.parquet"]
    dp = DataProcessor(
      train_data,
      "parquet",
      options = {
        "wt_name" : "wt",
      }
    )
    # Get number of effective events
    eff_events = dp.GetFull(
      method = "n_eff"
    )
    events_per_bin = 10000
    bins = min(int(np.ceil(eff_events/events_per_bin)),100)

    if self.verbose:
      print(f"- Number of bins for spline input: {bins}")

    nom_hist, bins = dp.GetFull(
      method = "histogram",
      column = self.parameter,
      bins = bins,
      extra_sel = "(classifier_truth == 0)",
    )
    shifted_hist, _ = dp.GetFull(
      method = "histogram",
      bins = bins,
      column = self.parameter,
      extra_sel = "(classifier_truth == 1)",
    )
    ratio = shifted_hist / nom_hist
    if self.verbose:
      print("- Normalising spline:")
      print(ratio)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    spline = CubicSpline(bin_centers, ratio, bc_type="clamped", extrapolate=True)

    def class_balancing(df, spline):
      df.loc[(df["classifier_truth"] == 0), "wt"] *= spline(df.loc[(df["classifier_truth"] == 0), self.parameter])
      return df.loc[:,["wt"]]

    for dt in ["train", self.test_name]:
      new_wt_name = f"wt_condition_normalised_{dt}.parquet"
      if os.path.isfile(f"{self.data_output}/{new_wt_name}"):
        os.system(f"rm {self.data_output}/{new_wt_name}")

      write_data = [f"{self.data_input}/X_{dt}.parquet", f"{self.data_input}/y_{dt}.parquet", f"{self.data_input}/wt_{dt}.parquet"]
      write_dp = DataProcessor(
        write_data,
        "parquet",
        options = {
          "wt_name" : "wt",
        }
      )
      write_dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(class_balancing, spline=spline),
          partial(self._WriteDataset, file_name=new_wt_name)
        ]
      )

  def Run(self):
    """
    Run the code utilising the worker classes
    """
    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the architecture in
    if self.verbose:
      print("- Loading in the architecture")
    with open(self.architecture, 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Normalise conditions to each other
    if self.verbose:
      print("- Normalise the conditions with a spline")
    self._NormaliseConditions()

    if self.initiate_wandb:

      if self.verbose:
        print("- Initialising wandb")

      r = RandomWords()
      wandb.init(project=self.wandb_project_name, name=f"{self.wandb_submit_name}_{r.get_random_word()}", config=architecture)
    
    # Build model
    if self.verbose:
      print("- Building the model")
    network = InitiateClassifierModel(
      architecture,
      self.data_input,
      test_name = self.test_name,
      wt_name = f"{self.data_output}/wt_condition_normalised",
      options = {
        "plot_dir" : self.plots_output,
        "disable_tqdm" : self.disable_tqdm,
        "use_wandb" : self.use_wandb,
        "data_parameters" : parameters["classifier"][self.parameter],
      }
    )

    network.BuildModel()
    network.BuildTrainer()

    # Do train
    if self.verbose:
      print("- Training the model")
    network.Train(name=f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}.h5")

    # Saving model architecture
    if self.verbose:
      print("- Saving the model and its architecture")
    architecture_save_name = f"{self.data_output}/{parameters['file_name']}{self.save_extra_name}_architecture.yaml"
    MakeDirectories(architecture_save_name)
    with open(architecture_save_name, 'w') as file:
      yaml.dump(architecture, file)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Add output wts from normalising spline
    outputs += [f"{self.data_output}/wt_condition_normalised_train.parquet"]
    outputs += [f"{self.data_output}/wt_condition_normalised_{self.test_name}.parquet"]

    # Add model
    outputs += [f"{self.data_output}/{self.file_name}{self.save_extra_name}.h5"]

    # Add architecture
    outputs += [f"{self.data_output}/{self.file_name}{self.save_extra_name}_architecture.yaml"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initialise inputs
    inputs = []

    # Add parameters
    inputs += [self.parameters]

    # Add architecture
    inputs += [self.architecture]

    # Add data input
    inputs += [
      f"{self.data_input}/X_train.parquet", 
      f"{self.data_input}/y_train.parquet",
      f"{self.data_input}/wt_train.parquet", 
      f"{self.data_input}/X_{self.test_name}.parquet",
      f"{self.data_input}/y_{self.test_name}.parquet", 
      f"{self.data_input}/wt_{self.test_name}.parquet",
    ]

    return inputs