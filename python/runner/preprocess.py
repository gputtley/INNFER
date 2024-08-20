import copy
import gc
import os
import pickle
import yaml

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from itertools import product
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split
from scipy.optimize import root_scalar

from data_processor import DataProcessor
from plotting import plot_histograms, plot_spline_and_thresholds
from useful_functions import GetYName, MakeDirectories, GetPOILoop, GetNuisanceLoop

class PreProcess():

  def __init__(self):
    """
    A class to preprocess the datasets and produce the data 
    parameters yaml file as well as the train, test and 
    validation datasets.
    """
    # Required input which is the location of a file
    self.cfg = None

    # Required input which can just be parsed
    self.file_name = None
    self.parquet_file_name = None

    # Other
    self.number_of_shuffles = 100
    self.split_validation_files = False
    self.verbose = True
    self.data_output = "data/"
    self.plots_output = "plots/"

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
    # Open the config
    with open(self.cfg, 'r') as yaml_file:
      cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Check if we are splitting nuisance model
    split_nuisances = False
    if "split_nuisance_models" in cfg.keys():
      split_nuisances = cfg["split_nuisance_models"]
      
    # Set up split nuisance loop
    if not split_nuisances:
      split_nuisance_loop = {
        self.file_name : {
          "selection" : cfg["preprocess"]["selection"] if "selection" in cfg["preprocess"].keys() else None,
          "columns" : cfg["variables"] + cfg["pois"] + cfg["nuisances"] + ["wt"],
          "data_output" : self.data_output,
        }
      }
    else:
      # Make nominal loop
      nominal_selection = "&".join([f"({k}==0)" for k in cfg["nuisances"]])
      split_nuisance_loop = {
        self.file_name : {
          "selection" : f'({cfg["preprocess"]["selection"]})&({nominal_selection})' if "selection" in cfg["preprocess"].keys() else nominal_selection,
          "columns" : cfg["variables"] + cfg["pois"] + ["wt"],
          "data_output" : self.data_output,          
        }
      }
      # Make nuisance loops
      for nuisance in cfg["nuisances"]:
        tmp_dp = DataProcessor([[self.parquet_file_name]], "parquet")
        if nuisance not in tmp_dp.GetColumnNames(): continue
        del tmp_dp
        gc.collect()
        nuisance_selection = "&".join([f"({k}==0)" for k in cfg["nuisances"] if k != nuisance])
        split_nuisance_loop[f"{self.file_name}_{nuisance}"] = {
          "selection" : f'({cfg["preprocess"]["selection"]})&({nuisance_selection})' if "selection" in cfg["preprocess"].keys() else nuisance_selection,
          "columns" : cfg["variables"] + cfg["pois"] + [nuisance] + ["wt"],
          "data_output" : self.data_output.replace(self.file_name,f"{self.file_name}_{nuisance}") if self.file_name in self.data_output else f"{self.data_output}/{nuisance}"
        }
      
    # Loop through individual file or split nuisances
    for output_name, output_info in split_nuisance_loop.items():

      if self.verbose:
        print(f"- Making inputs for {output_name}")

      # Set data output
      self.data_output = output_info["data_output"]

      # Set up data processor
      dp = DataProcessor(
        [[self.parquet_file_name]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : output_info["selection"],
          "columns" : output_info["columns"],
        }
      )

      # Initiate parameters dictionary
      parameters = {
        "file_name" : output_name,
      }

      # Write X and Y to parameters
      if self.verbose:
        print("- Finding X and Y variables")
      parameters["X_columns"] = sorted(cfg["variables"])
      parameters["Y_columns"] = sorted([var for var in cfg["pois"] + cfg["nuisances"] if var in dp.GetColumnNames()])

      # Get unique values
      if self.verbose:
        print("- Finding unique values of columns")
      unique_values = dp.GetFull(method="unique")
      unique_y_values = {k : sorted(v) for k, v in unique_values.items() if k in parameters["Y_columns"]}

      # Find unique values in Y for validation
      if self.verbose:
        print("- Finding Y values to perform validation on")
      parameters["unique_Y_values"] = copy.deepcopy(unique_y_values)
      if "validation_y_vals" in cfg["preprocess"].keys():
        for y_key in cfg["preprocess"]["validation_y_vals"].keys():
          if y_key in parameters["Y_columns"]:
            parameters["unique_Y_values"][y_key] = cfg["preprocess"]["validation_y_vals"][y_key]        

      # Find discrete variables in X to transform to continuous
      if self.verbose:
        print("- Finding discrete X variables and converting them to continuous ones")    
      discrete_X = {k : sorted(v) for k, v in unique_values.items() if k in parameters["X_columns"] and v is not None}
      if len(list(discrete_X.keys())) > 0:
        parameters["discrete_thresholds"] = {}
        parameters["spline_locations"] = {}
      for k, v in discrete_X.items():
        hist, _ = dp.GetFull(method="histogram", bins=v+[(2*v[-1])-v[-2]], column=k)
        pdf_spline, intervals, pre_integral = self._FitSplineAndFindIntervals(hist, v)
        spline_loc = f"{self.data_output}/spline_{k}.pkl"
        MakeDirectories(spline_loc)
        with open(spline_loc, 'wb') as file:
          pickle.dump(pdf_spline, file)
        parameters["discrete_thresholds"][k] = intervals
        parameters["spline_locations"][k] = spline_loc
        plot_spline_and_thresholds(pdf_spline, intervals, v, hist/pre_integral, x_label=k, name=f"{self.plots_output}/spline_for_{k}")

      # Making useful information
      self.dataset_loop = ["train","test","val"] if cfg["preprocess"]["train_test_val_split"].count(":") == 2 else ["train", "val"]
      train_test_y_vals = cfg["preprocess"]["train_test_y_vals"] if "train_test_y_vals" in cfg["preprocess"].keys() else {}
      validation_y_vals = cfg["preprocess"]["validation_y_vals"] if "validation_y_vals" in cfg["preprocess"].keys() else {}

      # Get sum of weights for each unique y combination
      if self.verbose:
        print("- Finding the total yields of entries for specific Y values and writing to file")
      if len(parameters["Y_columns"]) != 0:
        unique_y_combinations = list(product(*unique_y_values.values()))
        unique_combinations = pd.DataFrame(unique_y_combinations, columns=parameters["Y_columns"], dtype=np.float64)
        yield_df = copy.deepcopy(unique_combinations)
        yield_df.loc[:,"yield"] = dp.GetFull(method="sum_w_unique_columns", unique_combinations=unique_combinations).loc[:,"sum_w"]
        non_zero_yields = yield_df.loc[:,"yield"] != 0.0
        unique_combinations = unique_combinations.loc[non_zero_yields, :]
        yield_df = yield_df.loc[non_zero_yields, :]
        yield_df.loc[:,"effective_events"] = dp.GetFull(method="n_eff_unique_columns", unique_combinations=unique_combinations).loc[:,"eff_events"]
      else:
        yield_df = pd.DataFrame([[dp.GetFull(method="sum"), dp.GetFull(method="n_eff")]], columns=["yield","effective_events"], dtype=np.float64)

      # Get standardisation parameters
      if self.verbose:
        print("- Finding standardisation parameters")
      parameters["standardisation"] = {}
      means = dp.GetFull(
        method="mean",
        functions_to_apply = [
          partial(self._DoTrainTestValSplit, split="train", train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)
        ]
      )
      stds = dp.GetFull(
        method="std",
        functions_to_apply = [
          partial(self._DoTrainTestValSplit, split="train", train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)
        ]
      )    
      for k, v in means.items():
        parameters["standardisation"][k] = {"mean": v, "std": stds[k]}

      # Get yields in each one train test val split, this is used for normalisation later
      if self.verbose:
        if cfg["preprocess"]["train_test_val_split"].count(":") == 2:
          print("- Finding the total yields of entries for specific Y values in every train/test/val splitting")
        else:
          print("- Finding the total yields of entries for specific Y values in both train and val splittings")

      for split in self.dataset_loop:
        if self.verbose:
          print(f" - For split {split}")
        if len(parameters["Y_columns"]) != 0:
          yield_df.loc[:,f"yields_{split}"] = dp.GetFull(
            method="sum_w_unique_columns", 
            functions_to_apply = [
              partial(self._DoTrainTestValSplit, split=split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)
            ],
            unique_combinations = unique_combinations,
          ).loc[:,"sum_w"]
          yield_df.loc[:, f"effective_events_{split}"] = dp.GetFull(
            method="n_eff_unique_columns", 
            functions_to_apply = [
              partial(self._DoTrainTestValSplit, split=split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)
            ],
            unique_combinations = unique_combinations,
          ).loc[:,"eff_events"]

        else:
          yield_df.loc[0,f"yields_{split}"] = dp.GetFull(method="sum", functions_to_apply = [partial(self._DoTrainTestValSplit, split=split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)])
          yield_df.loc[0,f"effective_events_{split}"] = dp.GetFull(
            method="n_eff", 
            functions_to_apply = [
              partial(self._DoTrainTestValSplit, split=split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)
            ]
          )

      # Load and write batches
      if self.verbose:
        if cfg["preprocess"]["train_test_val_split"].count(":") == 2:
          print("- Making X/Y/wt and train/test/val split datasets and writing them to file")
        else:
          print("- Making X/Y/wt and train/val split datasets and writing them to file")
      dp.parameters = parameters

      parameters["file_loc"] = self.data_output
      for data_split in self.dataset_loop:

        for i in ["X","Y","wt"]:
          name = f"{self.data_output}/{i}_{data_split}.parquet"
          MakeDirectories(name)
          if os.path.isfile(name):
            os.system(f"rm {name}")

        dp.GetFull(
          method = None,
          functions_to_apply = [
            partial(self._DoTrainTestValSplit, split=data_split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals),
            partial(self._DoEqualiseYWeights, yields=yield_df, Y_columns=parameters["Y_columns"], scale_to=1.0, yield_name=f"yields_{data_split}"),
            "transform",
            partial(self._DoWriteDatasets, X_columns=parameters["X_columns"], Y_columns=parameters["Y_columns"], data_split=data_split)
          ]
        )

      # Delete data processor
      del dp
      gc.collect()

      # Shuffle dataset
      if self.verbose:
        print("- Shuffling dataset")
      for data_split in self.dataset_loop:
        for i in ["X","Y","wt"]:
          name = f"{self.data_output}/{i}_{data_split}.parquet"
          shuffle_name = f"{self.data_output}/{i}_{data_split}_shuffled.parquet"
          if os.path.isfile(shuffle_name):
            os.system(f"rm {shuffle_name}")   
          shuffle_dp = DataProcessor([[name]],"parquet")
          for i in range(self.number_of_shuffles):
            shuffle_dp.GetFull(
              method = None,
              functions_to_apply = [
                partial(self._DoShuffleIteration, iteration=i, total_iterations=self.number_of_shuffles, seed=42, dataset_name=shuffle_name)
              ]
            )
          os.system(f"mv {shuffle_name} {name}")

      # Write yields file
      if self.verbose:
        print("- Writing yields yaml")
      parameters["yield_loc"] = f"{self.data_output}/yields.parquet"
      MakeDirectories(parameters["yield_loc"])
      table = pa.Table.from_pandas(yield_df, preserve_index=False)
      pq.write_table(table, parameters["yield_loc"], compression='snappy')
      print(f"Created {parameters['yield_loc']}")

      # Write parameters file
      if self.verbose:
        print("- Writing parameters yaml")
      with open(self.data_output+"/parameters.yaml", 'w') as yaml_file:
        yaml.dump(parameters, yaml_file, default_flow_style=False)  
      print(f"Created {self.data_output}/parameters.yaml")

  
 
  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    # Add parquet files
    for data_split in self.dataset_loop:
      for i in ["X","Y","wt"]:
        outputs.append(f"{self.data_output}/{i}_{data_split}.parquet")
    # Add parameters file
    outputs.append(f"{self.data_output}/parameters.yaml")
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [
      self.cfg
    ]
    return inputs
        
  def _FitSplineAndFindIntervals(self, hist, bins):

    # Fit pdf
    unnormalised_pdf_spline = UnivariateSpline(bins, hist, s=0, k=min(len(bins)-1,3))
    integral = unnormalised_pdf_spline.integral(bins[0],bins[-1])
    hist_for_normalised_spline = hist/integral
    pdf_spline = UnivariateSpline(bins, hist_for_normalised_spline, s=0, k=min(len(bins)-1,3))

    # Find threholds for where the pdf integrals equal the cdf values
    cdf = [np.sum(hist[:ind+1]) for ind in range(len(bins))]
    cdf /= cdf[-1]
    intervals = [float(bins[0])]
    for desired_integral in cdf[:-1]:
      integral_diff = lambda b, bins, desired_integral: pdf_spline.integral(bins[0], b) - desired_integral
      integral_diff = partial(integral_diff, bins=bins, desired_integral=desired_integral)
      result = root_scalar(integral_diff, bracket=[bins[0], bins[-1]], method='brentq')
      intervals.append(float(result.root))
    intervals.append(float(bins[-1]))
    
    thresholds = {float(bins[ind]) : [intervals[ind],intervals[ind+1]] for ind in range(len(bins))}

    return pdf_spline, thresholds, integral
  
  def _DoEqualiseYWeights(self, df, yields, Y_columns=[], scale_to=1.0, yield_name="yield"):
    if len(Y_columns) > 0:
      unique_rows = df.loc[:,Y_columns].drop_duplicates()
      for _, ur in unique_rows.iterrows():
        matching_row = (yields.loc[:,Y_columns] == ur).all(axis=1)
        sum_weights = float(yields.loc[matching_row,yield_name].iloc[0])
        matching_rows = (df.loc[:,Y_columns] == ur).all(axis=1)
        df.loc[matching_rows,"wt"] = (scale_to / sum_weights) * df.loc[matching_rows,"wt"]
    else:
      sum_weights = float(yields.loc[:,yield_name].iloc[0])
      df.loc[:,"wt"] = (scale_to / sum_weights) * df.loc[:,"wt"]      
    return df

  def _DoShuffleIteration(self, df, iteration=0, total_iterations=10, seed=42, dataset_name="dataset.parquet"):

    # Select indices
    iteration_indices = (np.random.default_rng(seed).integers(0, total_iterations, size=len(df)) == iteration)

    # Write to file
    table = pa.Table.from_pandas(df.loc[iteration_indices, :], preserve_index=False)
    if os.path.isfile(dataset_name):
      combined_table = pa.concat_tables([pq.read_table(dataset_name), table])
      pq.write_table(combined_table, dataset_name, compression='snappy')
    else:
      pq.write_table(table, dataset_name, compression='snappy')

    return df

  def _DoWriteDatasets(self, df, X_columns=[], Y_columns=[], data_split="train", extra_name=""):

    for data_type, columns in {"X":X_columns, "Y":Y_columns, "wt":["wt"]}.items():
      file_path = f"{self.data_output}{extra_name}/{data_type}_{data_split}.parquet"
      table = pa.Table.from_pandas(df.loc[:, sorted(columns)], preserve_index=False)
      if os.path.isfile(file_path):
        combined_table = pa.concat_tables([pq.read_table(file_path), table])
        pq.write_table(combined_table, file_path, compression='snappy')
      else:
        pq.write_table(table, file_path, compression='snappy')
    return df

  def _DoTrainTestValSplit(self, df, split="train", train_test_val_split="0.6:0.1:0.3", train_test_y_vals={}, validation_y_vals={}):

    # Check whether to do test/val split
    test_val_split = (train_test_val_split.count(":") == 2)

    # Do train/test/val split
    train_ratio = float(train_test_val_split.split(":")[0])
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
    if test_val_split:
      test_ratio = float(train_test_val_split.split(":")[1])
      val_ratio = float(train_test_val_split.split(":")[2])   
      val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
    else:
      val_df = copy.deepcopy(temp_df)

    # Move train/test values to validation if unused
    removed_train_df = None
    removed_test_df = None
    for k, v in train_test_y_vals.items():
      # do for train
      if k in train_df.columns:
        if removed_train_df is None:
          removed_train_df = train_df[~train_df[k].isin(v)]
        else:
          removed_train_df = pd.concat([removed_train_df, train_df[~train_df[k].isin(v)]])
        train_df = train_df[train_df[k].isin(v)]
      # do for test
      if test_val_split:
        if k in test_df.columns:
          if removed_test_df is None:
            removed_test_df = test_df[~test_df[k].isin(v)]
          else:
            removed_test_df = pd.concat([removed_test_df, test_df[~test_df[k].isin(v)]])
          test_df = test_df[test_df[k].isin(v)]
    if removed_train_df is not None:
      val_df = pd.concat([val_df, removed_train_df])
    if test_val_split:
      if removed_test_df is not None:
        val_df = pd.concat([val_df, removed_test_df])

    # Move validation values to train/test if unused
    removed_val_df = None
    for k, v in validation_y_vals.items():
      if k in val_df.columns:
        if removed_val_df is None:
          removed_val_df = val_df[~val_df[k].isin(v)]
        else:
          removed_val_df = pd.concat([removed_val_df, val_df[~val_df[k].isin(v)]])
        val_df = val_df[val_df[k].isin(v)]
    if removed_val_df is not None:
      if test_val_split:
        train_ratio = float(train_test_val_split.split(":")[0])
        test_ratio = float(train_test_val_split.split(":")[1])
        sum_ratio = train_ratio + test_ratio
        test_ratio /= sum_ratio
        train_add_df, test_add_df = train_test_split(removed_val_df, test_size=test_ratio, random_state=42)
        train_df = pd.concat([train_df, train_add_df])
        test_df = pd.concat([test_df, test_add_df])
      else:
        train_df = pd.concat([train_df, removed_val_df])

    # Return correct split
    if split == "train":
      return train_df    
    if split == "test":
      return test_df
    elif split == "val":
      return val_df