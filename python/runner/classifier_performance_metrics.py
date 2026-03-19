import os
import yaml
import numpy as np
import pandas as pd
from functools import partial
from data_processor import DataProcessor
# from histogram_metrics_classifier import HistogramMetricsClassifier
from histogram_metrics import HistogramMetrics
from multidim_metrics import MultiDimMetrics
from yields import Yields
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from useful_functions import (
    GetValidationLoop,
    InitiateClassifierModel,
    LoadConfig,
    MakeDirectories,
    SkipEmptyDataset,
    SkipNonDensity,
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

    self.save_extra_name = ""
    self.metrics_save_extra_name = ""
    self.type = "syst"

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
  
  def Configure(self, options):
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):

    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the model in
    if self.verbose:
      print("- Loding in the config")
    self.open_cfg = LoadConfig(self.cfg)

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
            "data_parameters": parameters['classifier'][self.parameter]
        },
        test_name="test"
    )

    # Load weights into model
    if self.verbose:
      print(f"- Loading the classifier model {classifier_model_name}")
    self.network.Load(name=f"{classifier_model_name}.h5")

    # Set up metrics dictionary
    self.metrics = {}

    # Get histogram metrics
    if self.do_histogram_metrics:
      if self.verbose:
        print("- Getting histogram metrics")
      self.DoHistogramMetrics()

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

  def _GetFiles(self, data_type):

    X_file = f"{self.file_loc}/X_{data_type}.parquet"
    Y_file = f"{self.file_loc}/y_{data_type}.parquet"
    WT_file = f"{self.file_loc}/wt_{data_type}.parquet"

    return X_file, Y_file, WT_file

  def DoHistogramMetrics(self):

    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    def apply_classifier(df, func, X_columns):

      epsilon = 1e-7

      if "wt" not in df.columns:
        df.loc[:, "wt"] = 1.0
      df.loc[:, "wt_shift"] = 1.0
      probs = func(df.loc[:, X_columns])
      inds = (df["classifier_truth"] == 0)

      probs_0 = np.clip(probs[inds, 0], epsilon, 1 - epsilon)
      probs_1 = np.clip(probs[inds, 1], epsilon, 1 - epsilon)
      df.loc[inds, "wt_shift"] = probs_1 / probs_0

      df.loc[:, "probs"] = probs[:, 1]
      df.loc[:, "wt_total"] = df["wt"] * df["wt_shift"]

      return df.loc[inds].copy()

    classifier_model_name = f"{self.model_input}/{self.extra_model_dir}/{self.file_name}{self.save_extra_name}"

    with open(f"{classifier_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

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

    for data_type in self.histogram_datasets:

      x, y, wt = self._GetFiles(data_type)

      if self.verbose:
        print(f" - Doing histogram metrics for {data_type}")
      pred_df = DataProcessor(
          [[f"{parameters['classifier'][self.parameter]['file_loc']}/{i}_{data_type}.parquet" for i in ["X", "y", "wt"]]],
          "parquet",
          options={
              "parameters": parameters['classifier'][self.parameter],
          },
      )

      # Path to the Parquet file that _WriteDataset will create
      parquet_file = f"{self.data_output}/pred_{data_type}.parquet"

      # Only run GetFull if the file does not exist
      if not os.path.isfile(parquet_file):
        print("Parquet file not found. Running pred_df.GetFull() to create it...")
        pred_df.GetFull(
            method=None,
            functions_to_apply=[
                partial(
                    apply_classifier,
                    func=self.network.Predict,
                    X_columns=parameters['classifier'][self.parameter]["X_columns"],
                ),
                partial(self._WriteDataset, file_name=f"pred_{data_type}.parquet")
            ]
        )
      else:
        print(
            f"Parquet file already exists: {parquet_file}. Skipping GetFull().")

      hm = HistogramMetrics(
          [x, y, wt],
          [parquet_file],
          self.open_cfg["variables"],
          synth_wt_name="wt_total",
          sim_selection="classifier_truth==1"
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

      # Get total values
      if not self.do_chi_squared:
        continue
      count_chi_squared_per_dof = 0
      chi_squared_per_dof_total = 0
      for _, val_info in enumerate(
          GetValidationLoop(
              self.open_cfg, self.file_name)):
        if SkipNonDensity(
                self.open_cfg,
                self.file_name,
                val_info,
                skip_non_density=True):
          continue
        if SkipEmptyDataset(
                self.open_cfg,
                self.file_name,
                data_type,
                val_info):
          continue
        key = f"chi_squared_per_dof_{data_type}"
        if key not in self.metrics:
          continue
        chi_squared_per_dof_total += self.metrics[key]["sum"]
        count_chi_squared_per_dof += len(self.open_cfg["variables"])

      self.metrics[f"chi_squared_per_dof_{data_type}_sum"] = chi_squared_per_dof_total
      if count_chi_squared_per_dof > 0:
        self.metrics[f"chi_squared_per_dof_{data_type}_mean"] = (
            chi_squared_per_dof_total / count_chi_squared_per_dof
        )

      if self.do_kl_divergence:
        count_kl_divergence = 0
        kl_divergence_total = 0
        for _, val_info in enumerate(
            GetValidationLoop(
                self.open_cfg, self.file_name)):
          if SkipNonDensity(
                  self.open_cfg,
                  self.file_name,
                  val_info,
                  skip_non_density=True):
            continue
          if SkipEmptyDataset(
                  self.open_cfg,
                  self.file_name,
                  data_type,
                  val_info):
            continue
          if f"kl_divergence_{data_type}" not in self.metrics:
            continue
          kl_divergence_total += self.metrics[f"kl_divergence_{data_type}"]["sum"]
          count_kl_divergence += len(self.open_cfg["variables"])
        self.metrics[f"kl_divergence_{data_type}_sum"] = kl_divergence_total
        if count_kl_divergence > 0:
          self.metrics[f"kl_divergence_{data_type}_mean"] = kl_divergence_total / \
              count_kl_divergence

  def Outputs(self):
    # Load config
    cfg = LoadConfig(self.cfg)

    # Add metrics
    outputs = [
        f"{self.data_output}/metrics{self.save_extra_name}{self.metrics_save_extra_name}.yaml"]

    return outputs

  def Inputs(self):

    inputs = []

    # Add config
    inputs += [self.cfg]

    # Open config
    cfg = LoadConfig(self.cfg)

    # Add classifer model
    classifier_model_name = f"{self.model_input}/{self.extra_model_dir}/{self.file_name}{self.save_extra_name}"
    classifier_model_name = classifier_model_name.replace("//", "/")
    inputs += [f"{classifier_model_name}.h5"]
    inputs += [f"{classifier_model_name}_architecture.yaml"]

    # Add data
    if self.do_loss:
      for data_type in self.loss_datasets:
        inputs += [f"{self.file_loc}/X_{data_type}.parquet"]
        inputs += [f"{self.file_loc}/y_{data_type}.parquet"]
        inputs += [f"{self.file_loc}/wt_{data_type}.parquet"]

    datasets = []
    if self.do_histogram_metrics:
      datasets += self.histogram_datasets
    datasets = list(set(datasets))

    for data_type in datasets:
      # Add input files
      inputs += [f"{self.file_loc}/{i}_{data_type}.parquet" for i in ["X", "y", "wt"]]

    return inputs
