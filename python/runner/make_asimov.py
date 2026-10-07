import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import yaml
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from pandas.errors import PerformanceWarning
from scipy.interpolate import CubicSpline

from data_processor import DataProcessor
from data_loader import DataLoader
from density_correction import DensityCorrection
from useful_functions import InitiateClassifierModel, InitiateDensityModel, InitiateRegressionModel, MakeDirectories, LoadConfig, GetDefaultsInModel
from yields import Yields
from write_parquet import WriteParquet

warnings.simplefilter("ignore", PerformanceWarning)

class MakeAsimov():

  def __init__(self):

    self.cfg = None

    self.file_name = None
    self.density_model = None
    self.regression_models = None
    self.classifier_models = None
    self.model_extra_name = ""
    self.parameters = None
    self.model_input = "data/"
    self.data_output = "data/"
    self.n_asimov_events = 10**7
    self.seed = 42
    self.val_info = {}
    self.only_density = False
    self.add_truth = False
    self.scale_to_one = False
    self.extra_density_model_name = ""
    self.extra_regression_model_name = ""
    self.extra_classifier_model_name = ""
    self.use_asimov_scaling = None
    self.prune_classifier_models = None
    self.classifier_pruning_files = {}
    self.verbose = True
    self.skip_spline = True
    self.classifier_divide_by_nominal = False
    self.scale_up = 1.2
    self.drop_wt = False
    self.density_correction = False
    self.density_correction_model = None
    self.density_correction_parameters = None
    self.asimov_weights = None


  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)


  def Run(self):

    # Import costly packages
    import tensorflow as tf
    
    # Open cfg
    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg)

    # Open parameters
    if self.verbose:
      print("- Loading in parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Make all parameters
    if self.verbose:
      print("- Find model parameters")
    model_parameters = GetDefaultsInModel(parameters["file_name"], cfg)
    for lnN in parameters["yields"]["lnN"].keys():
      model_parameters[lnN] = 0.0
    for key, val in self.val_info.items():
      model_parameters[key] = val

    # Find yield
    if self.verbose:
      print("- Calculate yields prediction")
    yield_class = Yields(
      parameters["yields"]["nominal"],
      lnN = parameters["yields"]["lnN"],
      physics_model = None,
      rate_param = f"mu_{parameters['file_name']}" if f"mu_{parameters['file_name']}" in model_parameters.keys() else None,
    )

    if self.scale_to_one:
      total_yield = 1.0
    else:
      total_yield = yield_class.GetYield(pd.DataFrame({k:[v] for k,v in model_parameters.items()}))

    # Build the density model
    if self.verbose:
      print("- Building density network")

    density_model_name = f"{self.model_input}/{self.density_model['name']}{self.extra_density_model_name}/{parameters['file_name']}{self.model_extra_name}"

    with open(f"{density_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    network = InitiateDensityModel(
      architecture,
      self.density_model['file_loc'],
      options = {
        "data_parameters" : parameters["density"],
        "file_name" : self.file_name,
      }
    )
  
    # Loading density model
    if self.verbose:
      print(f"- Loading the density model {density_model_name}")
    network.Load(name=f"{density_model_name}.h5")

    # Correct the nominal density before applying nuisance model weights.
    correction = None
    if self.density_correction:
      if self.drop_wt:
        raise ValueError("Density correction weights cannot be dropped")
      correction = self._GetDensityCorrection()
      correction.Configure({
        "density_model": density_model_name,
        "density_parameters": parameters["density"],
      })
      correction.Load()

    if self.use_asimov_scaling is None:
      n_events_before = self.n_asimov_events
    else:
      n_events_before = int(np.ceil(total_yield*self.use_asimov_scaling))

    n_events = int(np.ceil(n_events_before*self.scale_up))

    # Sample from density model
    if self.verbose:
      print(f"- Sampling from density network")
    Y = pd.DataFrame({k:[v] for k,v in model_parameters.items() if k in parameters["density"]["Y_columns"]})

    asimov_writer = DataProcessor(
      [[partial(network.Sample, Y)]],
      "generator",
      n_events = n_events,
      wt_name = "wt",
      options = {
        "parameters" : parameters["density"],
        "scale" : total_yield,
      }
    )

    def add_truth(df, Y):
      return df.assign(**Y)

    functions_to_apply = []

    if correction is not None:
      def apply_density_correction(df):
        df["wt"] = df["wt"] * correction.Predict(df, model_parameters)
        return df
      functions_to_apply += [apply_density_correction]

    if self.add_truth:
      functions_to_apply += [partial(add_truth, Y=model_parameters)]

    wp = WriteParquet(name="asimov", data_output=self.data_output)    
    functions_to_apply += [wp]
    asimov_file_name = f"{self.data_output}/asimov.parquet"

    tf.random.set_seed(self.seed)
    tf.keras.utils.set_random_seed(self.seed)
    asimov_writer.GetFull(
      method = None,
      functions_to_apply = functions_to_apply
    )
    wp.collect()


    if not self.only_density:

      # Do regression models
      for regression_model in self.regression_models:

        # Build the regression model
        if self.verbose:
          print(f"- Building regresson network for {regression_model['parameter']}")
        regression_model_name = f"{self.model_input}/{regression_model['name']}{self.extra_regression_model_name}/{parameters['file_name']}"
        with open(f"{regression_model_name}_architecture.yaml", 'r') as yaml_file:
          architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

        network = InitiateRegressionModel(
          architecture,
          regression_model['file_loc'],
          options = {
            "data_parameters" : parameters['regression'][regression_model['parameter']]    
          }
        )  
      
        # Loading regression model
        if self.verbose:
          print(f"- Loading the regression model {regression_model_name}")
        network.Load(name=f"{regression_model_name}.h5")

        # Apply weights
        wt_shifter = DataProcessor(
          [[asimov_file_name]],
          "parquet",
          wt_name = "wt",
          options = {
          }
        )

        # Open normalising spline
        spline_name = f"{regression_model_name}_norm_spline.pkl"
        if not self.skip_spline:
          if self.verbose:
            print(f"- Loading the normalising spline for {regression_model_name}")
          with open(spline_name, 'rb') as f:
            spl = pickle.load(f)
        else:
          spl = None

        def apply_regression(df, func, X_columns, add_columns={}, spl=None, parameter=None):
          cols_in = list(df.columns)
          for k,v in add_columns.items(): df.loc[:,k] = v
          df["wt"] = df["wt"] * func(df.loc[:,X_columns]).flatten()
          if spl is not None:
            df["wt"] = df["wt"] * spl(df.loc[:,parameter]).flatten()
          return df.loc[:,cols_in]

        wt_shifter_name = f"asimov_wt_shifter_regression_{regression_model['parameter']}"
        total_wt_shifter_name = f"{self.data_output}/{wt_shifter_name}.parquet"
        wp = WriteParquet(name=wt_shifter_name, data_output=self.data_output)
        wt_shifter.GetFull(
          method = None,
          functions_to_apply = [
            partial(
              apply_regression, 
              func=network.Predict, 
              X_columns=parameters['regression'][regression_model['parameter']]["X_columns"],
              add_columns={regression_model['parameter']: model_parameters[regression_model['parameter']]},
              spl = spl,
              parameter = regression_model['parameter']
            ),
            wp
          ]
        )
        wp.collect()
        if os.path.isfile(total_wt_shifter_name): os.system(f"mv {total_wt_shifter_name} {asimov_file_name}")


      # Do classifier models
      for classifier_model in self._GetClassifierModels(model_parameters):

        # Check if we need to prune this model
        prune = False
        if self.prune_classifier_models is not None:
          with open(self.classifier_pruning_files[classifier_model['parameter']], 'r') as yaml_file:
            pruning_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
          prune = True
          for pruning_key, pruning_val in self.prune_classifier_models.items():
            if pruning_info[pruning_key] > pruning_val:
              prune = False
        if prune: 
          if self.verbose:
            print(f"- Pruning classifier model for model {classifier_model['name']}, parameter {classifier_model['parameter']}")
          continue

        # Build the classifier model
        if self.verbose:
          print(f"- Building classifier network for {classifier_model['parameter']}")
        classifier_model_name = f"{self.model_input}/{classifier_model['name']}{self.extra_classifier_model_name}/{parameters['file_name']}"
        with open(f"{classifier_model_name}_architecture.yaml", 'r') as yaml_file:
          architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

        network = InitiateClassifierModel(
          architecture,
          classifier_model['file_loc'],
          options = {
            "data_parameters" : parameters['classifier'][classifier_model['parameter']]
          }
        )

        # Loading classifier model
        if self.verbose:
          print(f"- Loading the classifier model {classifier_model_name}")
        network.Load(name=f"{classifier_model_name}.h5")

        # Apply weights
        wt_shifter = DataProcessor(
          [[asimov_file_name]],
          "parquet",
          wt_name = "wt",
          options = {
          }
        )

        # Open normalising spline
        spline_name = f"{classifier_model_name}_norm_spline.pkl"
        if not self.skip_spline:
          if self.verbose:
            print(f"- Loading the normalising spline for {classifier_model_name}")
          with open(spline_name, 'rb') as f:
            spl = pickle.load(f)
        else:
          spl = None

        def apply_classifier(df, func, X_columns, add_columns={}, spl=None, parameter=None, divide_by_nominal=False, nominal_columns={}):

          cols_in = list(df.columns)
          for k,v in add_columns.items(): df.loc[:,k] = v
          probs = func(df.loc[:,X_columns])

          if np.any(probs[:,0] == 0):
            zero_indices = np.where(np.isclose(probs[:, 0], 0.0))[0]
            probs[zero_indices,0] = 1
            probs[zero_indices,1] = 0
          df["wt"] = df["wt"] * probs[:,1] / probs[:,0]

          if divide_by_nominal:
            copy_df = df.copy()
            for k,v in nominal_columns.items(): copy_df.loc[:,k] = v
            func_eval = func(copy_df.loc[:,X_columns])
            if np.any(func_eval[:,0] == 0):
              zero_indices = np.where(np.isclose(func_eval[:, 0], 0.0))[0]
              func_eval[zero_indices,0] = 1
              func_eval[zero_indices,1] = 0
            nominal_probs = func_eval[:,1]/func_eval[:,0]

            #df["wt"] = df["wt"] / (nominal_probs)
            df["wt"] = np.divide(df["wt"], nominal_probs, out=np.zeros(len(df)), where=~np.isclose(nominal_probs, 0))

          if spl is not None:
            df["wt"] = df["wt"] * spl(df.loc[:,parameter]).flatten()

          return df.loc[:,cols_in]

        wt_shifter_name = f"asimov_wt_shifter_classifier_{classifier_model['parameter']}"
        total_wt_shifter_name = f"{self.data_output}/{wt_shifter_name}.parquet"
        wp = WriteParquet(name=wt_shifter_name, data_output=self.data_output)
        wt_shifter.GetFull(
          method = None,
          functions_to_apply = [
            partial(
              apply_classifier, 
              func=network.Predict, 
              X_columns=parameters['classifier'][classifier_model['parameter']]["X_columns"],
              add_columns={classifier_model['parameter']: model_parameters[classifier_model['parameter']]},
              spl = spl,
              parameter = classifier_model['parameter'],
              divide_by_nominal = self.classifier_divide_by_nominal,
              nominal_columns = {classifier_model['parameter']: 0.0}
            ),
            wp
          ]
        )
        wp.collect()
        if os.path.isfile(total_wt_shifter_name): os.system(f"mv {total_wt_shifter_name} {asimov_file_name}")


    # Select the first n_events_before events from the asimov dataset
    trim_dps = DataProcessor(
      [[asimov_file_name]],
      "parquet"
    )
    count_after = trim_dps.GetFull(method="count")
    if count_after < n_events_before:
      raise ValueError(f"Not enough events in the asimov dataset: {count_after} available, {n_events_before} required")
    elif count_after > n_events_before:
      if self.verbose:
        print(f"- Trimming the asimov dataset to the first {n_events_before} events")
      class counter:
        def __init__(self, n_events_before):
          self.n_events_before = n_events_before
          self.count = 0
        def __call__(self, df):
          if self.count == self.n_events_before:
            return df.iloc[:0, :]
          len_batch = len(df)
          if self.count + len_batch > self.n_events_before:
            df = df.iloc[:self.n_events_before - self.count, :]
          self.count += len(df)
          return df
      trim_counter = counter(n_events_before)
      trim_name = "asimov_first_n_events"
      wp = WriteParquet(name=trim_name, data_output=self.data_output)
      trim_dps.GetFull(
        method = None,
        functions_to_apply = [
          trim_counter,
          wp
        ]
      )
      wp.collect()
      if os.path.isfile(f"{self.data_output}/{trim_name}.parquet"): os.system(f"mv {self.data_output}/{trim_name}.parquet {asimov_file_name}")

    # PValue comparisons retain the original signed simulation weights, with
    # the learned density ratio multiplying them rather than being replaced.
    if self.asimov_weights is not None:
      weighted_dps = DataProcessor([[asimov_file_name]], "parquet")
      weights = DataLoader(self.asimov_weights, batch_size=weighted_dps.batch_size)
      if weights.num_rows != n_events_before:
        raise ValueError("Asimov and simulation weights must have the same number of rows")
      weighted_name = "asimov_simulation_weights"
      wp = WriteParquet(name=weighted_name, data_output=self.data_output)
      def apply_simulation_weights(df):
        batch_weights = weights.LoadNextBatch()["wt"].to_numpy(dtype=np.float64)
        if len(batch_weights) != len(df) or not np.all(np.isfinite(batch_weights)):
          raise ValueError("Misaligned or non-finite simulation weights")
        df["wt"] = df["wt"] * batch_weights
        return df
      try:
        weighted_dps.GetFull(method=None, functions_to_apply=[apply_simulation_weights, wp])
        wp.collect(memory_safe=True)
      finally:
        if weights.parquet_file is not None:
          weights.parquet_file.close()
      os.replace(f"{self.data_output}/{weighted_name}.parquet", asimov_file_name)

    # Rescale back to total yield
    if self.verbose:
      print(f"- Rescaling asimov dataset to total yield")
    wt_rescaler_name = f"asimov_wt_rescaler"
    total_wt_rescaler_name = f"{self.data_output}/{wt_rescaler_name}.parquet"
    wp = WriteParquet(name=wt_rescaler_name, data_output=self.data_output)
    wt_rescaler= DataProcessor(
      [[asimov_file_name]],
      "parquet",
      wt_name = "wt",
      options = {
      }
    )
    sum_wt = wt_rescaler.GetFull(method="sum")
    if not np.isfinite(sum_wt) or sum_wt <= 0:
      raise ValueError("Asimov weights must have a finite positive sum")
    def rescale_wt(df, scale):
      df["wt"] = df["wt"] * scale
      if self.drop_wt:
        df = df.drop(columns=["wt"])
      return df
    wt_rescaler.GetFull(
      method = None,
      functions_to_apply = [partial(rescale_wt, scale=total_yield/sum_wt), wp]
    )
    wp.collect()
    if os.path.isfile(total_wt_rescaler_name): os.system(f"mv {total_wt_rescaler_name} {asimov_file_name}")

    if self.density_correction:
      with open(f"{self.data_output}/density_correction.yaml", 'w') as file:
        yaml.safe_dump({
          "model": self.density_correction_model,
          "parameters": self.density_correction_parameters,
          "asimov_weights": self.asimov_weights,
        }, file)
    elif os.path.isfile(f"{self.data_output}/density_correction.yaml"):
      os.remove(f"{self.data_output}/density_correction.yaml")

    # print the total event count
    if self.verbose:
      final_dps = DataProcessor(
        [[asimov_file_name]],
        "parquet"
      )
      final_count = final_dps.GetFull(method="count")
      print(f"- Total event count in the asimov dataset: {final_count}")

    if self.verbose:
      print(f"- Finished making the asimov dataset: {asimov_file_name}")


  def Outputs(self):

    # Add asimov
    outputs = [f"{self.data_output}/asimov.parquet"]
    if self.density_correction:
      outputs += [f"{self.data_output}/density_correction.yaml"]

    return outputs


  def _GetDensityCorrection(self):
    """
    Set up the correction shared by generation and input declarations.
    """
    correction = DensityCorrection()
    correction.Configure({
      "model_input": self.density_correction_model,
      "parameters": self.density_correction_parameters,
    })
    return correction

  def _GetClassifierModels(self, model_parameters=None):
    """
    Select classifier models that can change the nominal-divided prediction.
    """
    if not self.classifier_divide_by_nominal:
      return self.classifier_models

    if model_parameters is None:
      cfg = LoadConfig(self.cfg)
      model_parameters = {**GetDefaultsInModel(self.file_name, cfg), **self.val_info}

    return [model for model in self.classifier_models if model_parameters.get(model["parameter"], 0.0) != 0.0]


  def Inputs(self):

    # Initiate inputs
    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add parameters
    inputs += [self.parameters]

    if self.density_correction:
      inputs += self._GetDensityCorrection().Inputs()
    if self.asimov_weights is not None:
      inputs += [self.asimov_weights]

    # Add density model
    density_model_name = f"{self.model_input}/{self.density_model['name']}{self.extra_density_model_name}/{self.file_name}{self.model_extra_name}"
    inputs += [f"{density_model_name}_architecture.yaml", f"{density_model_name}.h5"]

    if not self.only_density:
      # Add regression models
      for regression_model in self.regression_models:
        inputs += [f"{self.model_input}/{regression_model['name']}/{self.file_name}_architecture.yaml"]
        inputs += [f"{self.model_input}/{regression_model['name']}/{self.file_name}.h5"]
        if not self.skip_spline:
          inputs += [f"{self.model_input}/{regression_model['name']}/{self.file_name}_norm_spline.pkl"]

      # Add classifier models
      for classifier_model in self._GetClassifierModels():
        inputs += [f"{self.model_input}/{classifier_model['name']}/{self.file_name}_architecture.yaml"]
        inputs += [f"{self.model_input}/{classifier_model['name']}/{self.file_name}.h5"]
        if not self.skip_spline:
          inputs += [f"{self.model_input}/{classifier_model['name']}/{self.file_name}_norm_spline.pkl"]
        if self.prune_classifier_models is not None:
          inputs += [self.classifier_pruning_files[classifier_model['parameter']]]

    return inputs
