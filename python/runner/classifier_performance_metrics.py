import os
import yaml
import numpy as np
import pandas as pd
from functools import partial
from data_processor import DataProcessor
from histogram_metrics import HistogramMetrics
from multidim_metrics import MultiDimMetrics
from write_parquet import WriteParquet
from useful_functions import (
    InitiateClassifierModel,
    LoadConfig,
    MakeDirectories,
)

class ClassifierPerformanceMetrics():

  def __init__(self):
    self.cfg = None

    self.parameters = None
    self.parameter = None

    self.model_input = "models/"
    self.file_type = None
    self.file_name = None
    self.file_loc = None
    self.extra_model_dir = ""
    self.data_output = "data/"
    self.verbose = True

    self.do_loss = True
    self.loss_datasets = ["train", "test"]

    self.do_histogram_metrics = True
    self.do_chi_squared = True
    self.do_kl_divergence = True
    self.histogram_datasets = ["train", "test"]

    self.do_multidimensional_dataset_metrics = False
    self.do_bdt_separation = True
    self.do_wasserstein = True
    self.do_sliced_wasserstein = True
    self.do_kmeans_chi_squared = False
    self.multidimensional_datasets = ["train","test"]

    self.save_extra_name = ""
    self.metrics_save_extra_name = ""

    self.open_parameters = None

  def Configure(self, options):
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):

    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      self.open_parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the model in
    if self.verbose:
      print("- Loading in the config")
    self.open_cfg = LoadConfig(self.cfg)

    # Get the model name
    classifier_model_name = f"{self.model_input}/{self.extra_model_dir}/{self.file_name}{self.save_extra_name}"

    # Load the architecture in
    if self.verbose:
      print("- Loading in the architecture")
    with open(f"{classifier_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build model
    if self.verbose:
      print("- Building the model")
    self.network = InitiateClassifierModel(
      architecture,
      self.file_loc,
      options={
          "data_parameters": self.open_parameters['classifier'][self.parameter]
      },
      test_name="test"
    )

    # Load weights into model
    if self.verbose:
      print(f"- Loading the classifier model {classifier_model_name}")
    self.network.Load(name=f"{classifier_model_name}.h5")

    # Make predictions
    for data_type in self._GetAllDatasets():
      x, y, wt = self._GetFiles(data_type)
      pred_df = DataProcessor(
          [[x, y, wt]],
          "parquet",
          options={
              "parameters": self.open_parameters['classifier'][self.parameter],
          },
      )
      parquet_pred_file = f"pred_{data_type}"
      wp = WriteParquet(
        name=parquet_pred_file,
        data_output=self.data_output,
      )
      pred_df.GetFull(
        method=None,
        functions_to_apply=[
          partial(
            self._ApplyClassifier,
            func=self.network.Predict,
            X_columns=self.open_parameters['classifier'][self.parameter]["X_columns"],
          ),
          wp,
        ]
      )
      wp.collect(memory_safe=True)

    # Set up metrics dictionary
    self.metrics = {}

    # Get loss values
    if self.do_loss:
      if self.verbose:
        print("- Getting loss values")
      self.DoLoss()

    # Get histogram metrics
    if self.do_histogram_metrics:
      if self.verbose:
        print("- Getting histogram metrics")
      self.DoHistogramMetrics()

    # Get multidimensional dataset metrics
    if self.do_multidimensional_dataset_metrics:
      if self.verbose:
        print("- Getting multidimensional dataset metrics")
      self.DoMultiDimensionalDatasetMetrics()

    # Write to yaml
    if self.verbose:
      print("- Writing metrics yaml")
    output_name = f"{self.data_output}/metrics{self.save_extra_name}{self.metrics_save_extra_name}.yaml"
    MakeDirectories(output_name)
    with open(output_name, 'w') as yaml_file:
      yaml.dump(self.metrics, yaml_file, default_flow_style=False)

   # Print metrics
    for metric in sorted(list(self.metrics.keys())):
      if not isinstance(self.metrics[metric], dict):
        print(f"{metric} : {self.metrics[metric]}")
      else:
        print(f"{metric} :")
        for k1 in sorted(list(self.metrics[metric].keys())):
          if not isinstance(self.metrics[metric][k1], dict):
            print(f"  {k1} : {self.metrics[metric][k1]}")
          else:
            print(f"  {k1} :")
            for k2 in sorted(list(self.metrics[metric][k1].keys())):
              print(f"    {k2} : {self.metrics[metric][k1][k2]}")

  def _GetAllDatasets(self):
    data_types = []
    if self.do_loss:
      data_types += self.loss_datasets
    if self.do_histogram_metrics:
      data_types += self.histogram_datasets
    if self.do_multidimensional_dataset_metrics:
      data_types += self.multidimensional_datasets
    data_types = list(set(data_types))
    return data_types

  def _GetFiles(self, data_type):
    X_file = f"{self.file_loc}/X_{data_type}.parquet"
    Y_file = f"{self.file_loc}/y_{data_type}.parquet"
    WT_file = f"{self.file_loc}/wt_{data_type}.parquet"
    return X_file, Y_file, WT_file

  def _ApplyClassifier(self, df, func, X_columns, epsilon=1e-7):
    df.loc[:, "wt_shift"] = 1.0
    df.loc[:, "probs"] = 0.0
    inds = (df["classifier_truth"] == 0)
    probs = func(df.loc[inds, X_columns])
    probs_1 = np.clip(probs[:, 1], epsilon, 1 - epsilon)
    df.loc[inds, "wt_shift"] = probs_1 / (1-probs_1)
    df.loc[inds,"wt"] *= df.loc[:,"wt_shift"]
    df.loc[inds, "probs"] = probs[:, 1]
    return df.loc[:, ["wt_shift","wt"]]

  def DoHistogramMetrics(self):

    for data_type in self.histogram_datasets:

      if self.verbose:
        print(f" - Doing histogram metrics for {data_type}")

      # Load histogram metrics class
      x, y, wt = self._GetFiles(data_type)
      hm = HistogramMetrics(
          [x, y, wt],
          [x, y, f"{self.data_output}/pred_{data_type}.parquet"],
          self.open_cfg["variables"] + [self.parameter],
          sim_selection="classifier_truth==1",
          synth_selection="classifier_truth==0"
      )

      # Get chi squared values
      if self.do_chi_squared:
        if self.verbose:
          print(f" - Doing chi squared for {data_type} ")
        chi_squared, dof_for_chi_squared, chi_squared_per_dof = hm.GetChiSquared()
        if len(chi_squared.keys()) != 0:
          self.metrics[f"chi_squared_{data_type}"] = chi_squared
          self.metrics[f"dof_for_chi_squared_{data_type}"] = dof_for_chi_squared
          self.metrics[f"chi_squared_per_dof_{data_type}"] = chi_squared_per_dof

      # Get kl divergence values
      if self.do_kl_divergence:
        if self.verbose:
          print(f" - Doing kl divergence for {data_type}")
        kl_divergence = hm.GetKLDivergence()
        if len(kl_divergence.keys()) != 0:
          self.metrics[f"kl_divergence_{data_type}"] = kl_divergence

  def DoLoss(self):

    # Get loss values
    for data_type in self.loss_datasets:
      if self.verbose:
        print(f"  - Getting loss for {data_type}")
      self.metrics[f"loss_{data_type}"] = float(self.network._GetEpochLoss(data_type=data_type))
    if "train" in self.loss_datasets and "test" in self.loss_datasets:
      self.metrics["loss_difference"] = self.metrics["loss_test"] - self.metrics["loss_train"]
      self.metrics["loss_test_plus_difference"] = self.metrics["loss_test"] + self.metrics["loss_difference"]


  def DoMultiDimensionalDatasetMetrics(self):

    for data_type in self.multidimensional_datasets:
      if self.verbose:
        print(f"  - Getting multidimensional dataset metrics for {data_type}")

      x, y, wt = self._GetFiles(data_type)

      # Initialise multi metrics
      mm = MultiDimMetrics(
        [x, y, wt],
        [x, y, f"{self.data_output}/pred_{data_type}.parquet"],
        self.open_parameters['classifier'][self.parameter]["X_columns"],
        sim_selection="classifier_truth==1",
        synth_selection="classifier_truth==0"
      )
      mm.verbose = self.verbose

      # Get BDT separation metric
      if self.do_bdt_separation:
        if self.verbose:
          print(" - Adding BDT separation")
        mm.AddBDTSeparation()
      
      # Get Wasserstein metric
      if self.do_wasserstein:
        if self.verbose:
          print(" - Adding Wasserstein")
        mm.AddWassersteinUnbinned()

      # Get sliced Wasserstein metric
      if self.do_sliced_wasserstein:
        if self.verbose:
          print(" - Adding sliced Wasserstein")
        mm.AddWassersteinSliced()

      if self.do_kmeans_chi_squared:
        if self.verbose:
          print(" - Adding kmeans chi squared")
        mm.AddKMeansChiSquared()

      # Run metrics
      multidim_metrics = mm.Run()
      self.metrics = {**self.metrics, **{f"{k}_{data_type}": v for k, v in multidim_metrics.items()}}

  def Outputs(self):

    # Add metrics
    outputs = [f"{self.data_output}/metrics{self.save_extra_name}{self.metrics_save_extra_name}.yaml"]

    return outputs

  def Inputs(self):

    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add classifer model
    classifier_model_name = f"{self.model_input}/{self.extra_model_dir}/{self.file_name}{self.save_extra_name}"
    classifier_model_name = classifier_model_name.replace("//", "/")
    inputs += [f"{classifier_model_name}.h5"]
    inputs += [f"{classifier_model_name}_architecture.yaml"]

    # Add data
    for dataset in self._GetAllDatasets():
      x, y, wt = self._GetFiles(dataset)
      inputs += [x, y, wt]

    return inputs
