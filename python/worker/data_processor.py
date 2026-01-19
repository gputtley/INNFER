import time

import copy
import os
import pickle
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from useful_functions import CustomHistogram, Resample

class DataProcessor():

  def __init__(
      self, 
      datasets, 
      dataset_type, 
      wt_name = None, 
      n_events = None,
      batch_size = None,
      options = {}
    ):
    if dataset_type not in ["dataset","parquet","generator"]:
      raise ValueError("dataset_type given is incorrect.")
    else:
      self.dataset_type = dataset_type
    if dataset_type in ["generator"] and n_events is None:
      raise ValueError("Must specify n_events when using generators.")
    else:
      self.n_events = n_events

    # Make sure datasets has the correct shape
    if isinstance(datasets, list):
      if isinstance(datasets[0], list):
        self.datasets = datasets
      else:
        self.datasets = [datasets]
    else:
      self.datasets = [[datasets]]

    # Options to run
    self.wt_name = wt_name
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH")) if batch_size is None else batch_size
    self.selection = None
    self.columns = None
    self.drop_columns = None
    self.scale = None
    self.resample = False
    self.resample_drop_negative_weights = False
    self.resampling_seed = 1
    self.functions = []
    self.decimals = 16
    self.check_wt = False

    # Transform options
    self.parameters = {}

    # Set options
    self._SetOptions(options)

    # Calculate import features
    if dataset_type == "dataset":
      self.batch_size = max([len(ds[0]) for ds in self.datasets])
      self.num_batches = [1 for _ in datasets]
    elif dataset_type == "generator":
      self.num_batches = [int(np.ceil(n_events/self.batch_size)) for _ in self.datasets]
      self.n_events_per_batch = []
      for num_in_batch in self.num_batches:
        self.n_events_per_batch.append([self.batch_size]*(num_in_batch))
        remainder = n_events % self.batch_size
        if remainder != 0:
          self.n_events_per_batch[-1][-1] = remainder
    elif dataset_type == "parquet":
      self.data_loaders = [[DataLoader(d, batch_size=self.batch_size) for d in ds] for ds in self.datasets]
      self.num_batches = [self.data_loaders[ind][0].num_batches for ind in range(len(self.datasets))]

    #if np.sum(self.num_batches) == 0:
    #  print("WARNING: No batches found!")

    if not isinstance(self.scale, list):
      self.scale = [self.scale]*len(self.datasets)

    # Variables needed for running
    self.file_ind = 0
    self.batch_ind = 0
    self.finished = False
    self.total_columns = None
    self.sort_columns = True

    # Cache
    self.cache_to_memory = False
    self.dataset_cache = {}


  def _SetOptions(
      self, 
      options
    ):

    for key, value in options.items():
      setattr(self, key, value)

  
  def _ClearCacheAndSetting(self):
    self.dataset_cache = {}
    self.cache_to_memory = False


  def _ReplaceEqualsWithIsClose(self, selection):
    pattern = r'(\w+)\s*==\s*([-+]?\d*\.?\d+)'      
    replacement = r'np.isclose(\1, \2)'   
    shift = 1e-7
    replacement = rf'((\1 > (\2 - {shift})) & (\1 < (\2 + {shift})))'   
    modified_selection = re.sub(pattern, replacement, selection)
    return modified_selection


  def _AddFinishedCounters(self):
    # Change batch and file ind
    if self.batch_ind + 1 == self.num_batches[self.file_ind]:
      self.batch_ind = 0
      if self.file_ind + 1 == len(self.datasets):
        self.file_ind = 0
        self.finished = True
      else:
        self.file_ind += 1
    else:
      self.batch_ind += 1


  def Restart(
      self
    ):
    self.file_ind = 0
    self.batch_ind = 0
    self.finished = False

  def LoadNextBatch(
      self, 
      extra_sel = None, 
      functions_to_apply = [],
      print_dataset = False,
    ):

    functions_to_apply = self.functions + functions_to_apply

    self.finished = False
    df = None

    for column_ind in range(len(self.datasets[self.file_ind])):
    
      # Get dataset
      if self.dataset_type == "dataset":
        tmp = self.datasets[self.file_ind][column_ind]
      elif self.dataset_type == "generator":
        tmp = self.datasets[self.file_ind][column_ind](self.n_events_per_batch[self.file_ind][self.batch_ind])
      elif self.dataset_type == "parquet":
        tmp = self.data_loaders[self.file_ind][column_ind].LoadNextBatch()

      # Skip if no columns
      if len(tmp.columns) == 0: continue

      # Combine columns into a full batch
      if df is None:
        df = tmp.reindex(tmp.index, copy=False)
      else:
        #df = pd.concat([df, tmp.loc[:, ~tmp.columns.isin(df.columns)]], axis=1)
        new_cols = tmp.columns.difference(df.columns)
        df[new_cols] = tmp[new_cols].to_numpy()

    if df is None: 
      self.finished = True
      return None

    # Set data type
    if len(df) > 0 and len(df.columns) > 0:
      if not all(df.dtypes == "float"):
        df = df.astype("float", copy=False)

    # Scale weights
    if self.scale[self.file_ind] is not None:
      if self.wt_name is None: 
        self.wt_name = "wt"
      scale = self.scale[self.file_ind] if self.dataset_type != "generator" else self.scale[self.file_ind]/self.n_events  
      if self.wt_name in df.columns:
        df[self.wt_name] *= scale
      else:
        df[self.wt_name] = scale

    # Resample
    if self.resample and self.wt_name is not None:
      sum_wts = int(round(np.sum(df[self.wt_name].to_numpy()),0))
      if self.resample_drop_negative_weights:
        df = df.loc[df[self.wt_name] >= 0,:]
      columns_without_weights = list(df.columns)
      columns_without_weights.remove(self.wt_name)
      data, wts = Resample(df[columns_without_weights].to_numpy(), df[self.wt_name].to_numpy(), n_samples=sum_wts, seed=self.resampling_seed)
      df = pd.DataFrame(np.hstack((data,wts.reshape(-1,1))), columns=columns_without_weights+[self.wt_name], dtype=np.float64)

    # Apply functions
    if "transform" not in functions_to_apply and "untransform" not in functions_to_apply:
      if "selection" not in functions_to_apply:
        df = self.ApplySelection(df, extra_sel=extra_sel)
    for f in functions_to_apply:
      if isinstance(f, str):
        if f == "untransform":
          df = self.UnTransformData(df)
          if "selection" not in functions_to_apply:
            df = self.ApplySelection(df, extra_sel=extra_sel)
        elif f == "transform":
          if "selection" not in functions_to_apply:
            df = self.ApplySelection(df, extra_sel=extra_sel)
          df = self.TransformData(df)
        elif f == "selection":
          df = self.ApplySelection(df, extra_sel=extra_sel)
      else:
        df = f(df)

    # Check weight exists
    if self.wt_name not in list(df.columns) and self.check_wt:
      self.wt_name = None

    # Select the columns
    if self.columns is not None:
      df = df[[col for col in self.columns if col in df.columns]]
    if self.drop_columns is not None:
      df = df.drop(columns=[col for col in self.drop_columns if col in df.columns])

    # Sort columns
    if self.sort_columns:
      if self.total_columns is None:
        self.total_columns = sorted(list(df.columns))
      #df = df[self.total_columns]
      df = df.reindex(columns=self.total_columns)

    # Change batch and file ind
    self._AddFinishedCounters()


    return df


  def GetFull(
      self, 
      method = "dataset", 
      column = None,
      extra_sel = None,
      functions_to_apply = [],
      bins = 10,
      ignore_discrete = False,
      density = False, 
      unique_threshold = 40, 
      discrete_binning = True,
      quantile = 0.01,
      ignore_quantile = 0.002,
      unique_combinations = None,
      sampling_fraction = 1.0,
      means = None,
      custom = None,
      test_fraction = 0.2,
      custom_options = {},
      print_dataset = False,
      skip_load=False
    ):

    none_total_columns = False
    if self.total_columns is None:
      none_total_columns = True

    if method == "std": # Get the mean for standard deviation calculation
      if means is None:
        sum_cols, sum_wts = self.GetFull(method="sum_columns", extra_sel=extra_sel, functions_to_apply=functions_to_apply)
        means = {k : v/sum_wts for k, v in sum_cols.items()}
    elif method == "n_eff": # Get the number of effective events
      sum_wts_squared = self.GetFull(method="sum_w2", extra_sel=extra_sel, functions_to_apply=functions_to_apply)
      sum_wts = self.GetFull(method="sum", extra_sel=extra_sel, functions_to_apply=functions_to_apply)
      if sum_wts_squared != 0:
        return (sum_wts**2)/sum_wts_squared
      else:
        return 0
    elif method == "n_eff_unique_columns": # Get the number of effective events in unique columns
      sum_wts_squared = self.GetFull(method="sum_w2_unique_columns", extra_sel=extra_sel, functions_to_apply=functions_to_apply, unique_combinations=unique_combinations)
      sum_wts = self.GetFull(method="sum_w_unique_columns", extra_sel=extra_sel, functions_to_apply=functions_to_apply, unique_combinations=unique_combinations)
      eff_events = copy.deepcopy(sum_wts)
      eff_events["eff_events"] = eff_events["sum_w"]**2
      eff_events.drop(["sum_w"], axis=1, inplace=True)
      eff_events["eff_events"] = np.where(sum_wts_squared["sum_w2"] == 0, 0, eff_events["eff_events"] / sum_wts_squared["sum_w2"])
      return eff_events
    elif method == "bins_with_equal_spacing": # Get equally spaced bins
      if not ignore_discrete:
        unique = self.GetFull(method="unique", extra_sel=extra_sel, functions_to_apply=functions_to_apply, unique_threshold=bins)[column]
      else:
        unique = None
      if unique is not None: # Discrete bins
        unique = sorted(unique)
        return np.array(unique + [2*unique[-1] - unique[-2]])
      else:
        return np.linspace(self.GetFull(method="quantile", extra_sel=extra_sel, functions_to_apply=functions_to_apply, column=column, quantile=ignore_quantile), self.GetFull(method="quantile", extra_sel=extra_sel, functions_to_apply=functions_to_apply, column=column, quantile=1-ignore_quantile), num=bins+1)
    elif method == "bins_with_equal_stats": # Get equal stat bins
      unique = self.GetFull(method="unique", extra_sel=extra_sel, functions_to_apply=functions_to_apply, unique_threshold=bins)[column]
      if unique is not None and not ignore_discrete: # Discrete bins
        unique = sorted(unique)
        return np.array(unique + [2*unique[-1] - unique[-2]])
      else:
        return np.array([self.GetFull(method="quantile", extra_sel=extra_sel, functions_to_apply=functions_to_apply, column=column, quantile=i) for i in np.linspace(ignore_quantile, 1-ignore_quantile, num=bins+1)])
    elif method in ["histogram","histogram_and_uncert"]: # Get histogram bins
      if type(bins) == int:
        bins = self.GetFull(method="bins_with_equal_spacing", extra_sel=extra_sel, functions_to_apply=functions_to_apply, column=column, bins=bins, ignore_discrete=ignore_discrete)

    self.batch_ind = 0
    self.file_ind = 0
    load_ind = 0
    out = None

    while not self.finished:

      if skip_load:
        # Skip loading data
        tmp = None
      else:
        if self.cache_to_memory and (load_ind in self.dataset_cache.keys()):
          # Load from cache
          tmp = self.dataset_cache[load_ind]
          self._AddFinishedCounters()
        else:
          # Load batch
          tmp = self.LoadNextBatch(
            extra_sel = extra_sel,
            functions_to_apply = functions_to_apply,
            print_dataset = print_dataset,
          )
          if self.cache_to_memory:
            self.dataset_cache[load_ind] = tmp
        #if self.cache_to_memory:
        #  tmp = tmp.copy()      

      #if print_dataset:
      #  print(tmp)

      # Run method
      if method in ["dataset"]: # get full dataset
        out = self._method_dataset(tmp, out)
      elif method in ["sampled_dataset"]: # get sampled dataset
        out = self._method_sampled_dataset(tmp, out, sampling_fraction=sampling_fraction)
      elif method in ["train_test_split"]: # get train and test dataset
        out = self._method_train_test_split(tmp, out, column, test_fraction=test_fraction)
      elif method in ["histogram"]: # make a histogram from the dataset
        out = self._method_histogram(tmp, out, column, bins=bins, discrete_binning=discrete_binning)
      elif method in ["histogram_and_uncert"]: # make a histogram from the dataset
        out = self._method_histogram_and_uncert(tmp, out, column, bins=bins, discrete_binning=discrete_binning)
      elif method in ["histogram_2d"]: # make a histogram from the dataset
        out = self._method_histogram_2d(tmp, out, column, bins=bins)
      elif method in ["histogram_2d_and_uncert"]: # make a histogram from the dataset
        out = self._method_histogram_2d_and_uncert(tmp, out, column, bins=bins)
      elif method in ["sum"]: # sum up the weights
        out = self._method_sum(tmp, out, count=False)
      elif method in ["sum_w_unique_columns"]: # sum up the weights for unique values of columns
        out = self._method_sum_w_unique_columns(tmp, out, unique_combinations)
      elif method in ["count"]: # count events
        out = self._method_sum(tmp, out, count=True)
      elif method in ["count_unique_columns"]: # sum up the weights for unique values of columns
        out = self._method_count_unique_columns(tmp, out, unique_combinations)
      elif method in ["sum_w2"]: # sum up the weights or count events
        out = self._method_sum_w2(tmp, out)
      elif method in ["sum_w2_unique_columns"]: # sum up the weights for unique values of columns
        out = self._method_sum_w2_unique_columns(tmp, out, unique_combinations)
      elif method in ["sum_columns","mean"]: # find sum of columns - also used for the mean
        out = self._method_sum_columns(tmp, out)
      elif method in ["sum_formula"]:
        out = self._method_sum_formula(tmp, out, column)
      elif method in ["std"]: # find partial information for std of columns
        out = self._method_part_std(tmp, out, means)
      elif method in ["unique"]: # find unique values of a column
        out = self._method_unique(tmp, out, unique_threshold)
      elif method in ["quantile"]: # find a quantile of the dataset
        out = self._method_part_quantile(tmp, out, column, quantile)
      elif method in ["min_max"]: # find min and max of columns
        out = self._method_min_max(tmp, out)
      elif method in ["custom"]: # custom function
        out = custom(tmp, out, options=custom_options)
      else: 
        out = None

      load_ind += 1

      # Remove tmp from memory
      #del tmp

    self.finished = False

    if none_total_columns:
      self.total_columns = None

    if method == "mean": # calculate means
      return {k : v/out[1] for k, v in out[0].items()}
    elif method == "std": # calculate std
      return {k : float(np.sqrt(v/out[1])) for k, v in out[0].items()}
    elif method == "histogram" and density: # normalise histograms
      sum_wt = np.sum(out[0])
      return out[0]/sum_wt, out[1]
    elif method == "histogram_and_uncert" and density: # normalise histograms
      sum_wt = np.sum(out[0])
      return out[0]/sum_wt, out[1]/sum_wt, out[2]
    elif method == "quantile": # return only average quantile
      return out[0]

    return out


  def GetColumnNames(
      self
    ):
    df = self.LoadNextBatch()
    self.Restart()
    return list(df.columns)    


  def TransformData(self, data):
    """
    Transform columns in the dataset.

    Args:
        data (pd.DataFrame): The dataset to be transformed.

    Returns:
        pd.DataFrame: The transformed dataset.
    """
    for column_name in data.columns:

      # Apply standardisation
      if column_name in self.parameters["standardisation"]:
        data[column_name] = self.Standardise(data[column_name], column_name).astype(np.float64)
        
    return data

  def UnTransformData(
      self, 
      data
    ):
    """
    Untransform specified columns in the dataset.

    Args:
        data (pd.DataFrame): The dataset to be Untransform.

    Returns:
        pd.DataFrame: The Untransform dataset.
    """
    for column_name in data.columns:

      # Unstandardise the probabilities
      if column_name == "prob":
        for col in self.parameters["X_columns"]:
          data[column_name] /= self.parameters["standardisation"][col]["std"]
        if "discrete_integral_differences" in self.parameters.keys():
          for col in self.parameters["discrete_integral_differences"]:
            data[column_name] /= self.parameters["discrete_integral_differences"][col]

      # Unstandardise the log probabilities
      if column_name == "log_prob":
        for col in self.parameters["X_columns"]:
          data[column_name] -= np.log(self.parameters["standardisation"][col]["std"])
        if "discrete_integral_differences" in self.parameters.keys():
          for col in self.parameters["discrete_integral_differences"]:
            data[column_name] -= self.parameters["discrete_integral_differences"][col]

      # Unstandardise the first derivative of the log probabilities
      if column_name.startswith("d_log_prob_by_d_"):
        col_1 = column_name.split("d_log_prob_by_d_")[1]
        if col_1 in self.parameters["standardisation"].keys():
          data[column_name] /= self.parameters["standardisation"][col_1]["std"]

      # Unstandardise the second derivative of the log probabilities
      if column_name.startswith("d2_log_prob_by_d_"):
        col_1 = column_name.split("d2_log_prob_by_d_")[1].split("_and_")[0]
        col_2 = column_name.split("d2_log_prob_by_d_")[1].split("_and_")[1]
        if col_1 in self.parameters["standardisation"].keys() and col_2 in self.parameters["standardisation"].keys():
          data[column_name] /= (self.parameters["standardisation"][col_1]["std"] * self.parameters["standardisation"][col_2]["std"])

      # Unstandardise the first derivative of the one standardised paramater with respect to another
      for col_1 in self.parameters["standardisation"].keys():
        if column_name.startswith(f"d_{col_1}_by_d_"):
          for col_2 in self.parameters["standardisation"].keys():
            if column_name == f"d_{col_1}_by_d_{col_2}":
              data[column_name] *= self.parameters["standardisation"][col_1]["std"] / self.parameters["standardisation"][col_2]["std"]

      # Unstandardise the second derivative of the one standardised paramater with respect to another
      for col_1 in self.parameters["standardisation"].keys():
        if column_name.startswith(f"d2_{col_1}_by_d_"):
          for col_2 in self.parameters["standardisation"].keys():
            for col_3 in self.parameters["standardisation"].keys():
              if column_name == f"d2_{col_1}_by_d_{col_2}_and_{col_3}":
                data[column_name] *= self.parameters["standardisation"][col_1]["std"] / (self.parameters["standardisation"][col_2]["std"] * self.parameters["standardisation"][col_3]["std"])

      # Unstandardise columns
      if column_name in list(self.parameters["standardisation"].keys()):
        data[column_name] = self.UnStandardise(data[column_name], column_name)

    return data


  def Standardise(
      self, 
      column, 
      column_name
    ):
    """
    Standardise a column.

    Args:
        column (pd.Series): The column to be standardised.
        column_name (str): The name of the column.

    Returns:
        pd.Series: The standardised column.
    """
    return (column - self.parameters["standardisation"][column_name]["mean"])/self.parameters["standardisation"][column_name]["std"]
  

  def UnStandardise(
      self, 
      column, 
      column_name
    ):
    """
    Unstandardize a column.

    Args:
        column (pd.Series): The standardised column to be unstandardised.
        column_name (str): The name of the column.

    Returns:
        pd.Series: The unstandardised column.
    """
    if column_name in self.parameters["standardisation"]:
      return (column*self.parameters["standardisation"][column_name]["std"]) + self.parameters["standardisation"][column_name]["mean"]
    else:
      return column
    
  def DiscreteToContinuous(
      self,
      column,
      column_name,
      n_integral_bins = 100000
    ):
    # Open spline
    with open(self.parameters["spline_locations"][column_name], 'rb') as file:
      spline = pickle.load(file)

    column_indices = column.index
    column = column.to_numpy()
    for k, v in self.parameters["discrete_thresholds"][column_name].items():
      # Find matching indices
      indices = (column == k)
      n_samples = len(column[indices]) 

      # Compute the CDF
      param_values = np.linspace(v[0], v[1], n_integral_bins)
      cdf_vals = np.cumsum(np.abs(spline(param_values))) / np.sum(np.abs(spline(param_values)))

      # Normalise the CDF
      cdf_vals /= cdf_vals[-1]

      # Generate random numbers
      random_nums = np.random.rand(n_samples)

      # Inverse transform sampling
      column[indices] = np.interp(random_nums, cdf_vals, param_values)

    return pd.DataFrame({column_name : column.flatten()}, index=column_indices, dtype=np.float64)


  def UnDiscreteToContinuous(
      self,
      column,
      column_name,
    ):
    for k, v in self.parameters["discrete_thresholds"][column_name].items():

      # Find matching indices
      if k == min(list(self.parameters["discrete_thresholds"][column_name].keys())):
        indices = (column < v[1])
      elif k == max(list(self.parameters["discrete_thresholds"][column_name].keys())):
        indices = (column >= v[0])
      else:
        indices = ((column >= v[0])) & (column < v[1])

      # Do inverse
      column = column.copy()
      column.loc[indices] = k

    return column

  def ApplySelection(
      self, 
      data,
      extra_sel = None
    ):
    if self.selection is not None or extra_sel is not None:
      if extra_sel is None:
        selection = self.selection
      elif self.selection is None:
        selection = extra_sel
      else:
        selection = f"({self.selection}) & {extra_sel}"

      selection = self._ReplaceEqualsWithIsClose(selection)
      data = data.loc[data.eval(selection),:]
    return data

  def _method_count_unique_columns(self, tmp, out, unique_combinations, count=False):

    for ind, uc in unique_combinations.iterrows():

      selection = " & ".join([f"({column}=={float(unique_combinations.loc[ind,column])})" for column in list(unique_combinations.columns)])
      selection = self._ReplaceEqualsWithIsClose(selection)
      tmp_tmp = tmp.loc[tmp.eval(selection),:]
      tmp_total = len(tmp_tmp)

      if out is None:
        out = copy.deepcopy(unique_combinations)
        out["count"] = 0.0

      out.loc[ind, "count"] = float(out.loc[ind, "count"]) + tmp_total

    return out


  def _method_dataset(self, tmp, out):
    if out is None:
      out = tmp.reindex(tmp.index, copy=False)
    else:
      out = pd.concat([out, tmp], axis=0, ignore_index=True)
    return out


  def _method_histogram(self, tmp, out, column, bins=40, discrete_binning=False):
    if self.wt_name is None:
      tmp_hist, bins = CustomHistogram(tmp[column], bins=bins, discrete_binning=discrete_binning)
    else:
      tmp_hist, bins = CustomHistogram(tmp[column], weights=tmp[self.wt_name], bins=bins, discrete_binning=discrete_binning)

    if out is None:
      out = [copy.deepcopy(tmp_hist), copy.deepcopy(bins)]
    else:
      out[0] += tmp_hist
    return out


  def _method_histogram_and_uncert(self, tmp, out, column, bins=40, discrete_binning=False):
    if self.wt_name is None:
      tmp_hist, tmp_hist_uncert, bins = CustomHistogram(tmp[column], bins=bins, discrete_binning=discrete_binning, add_uncert=True)
    else:
      tmp_hist, tmp_hist_uncert, bins = CustomHistogram(tmp[column], weights=tmp[self.wt_name], bins=bins, discrete_binning=discrete_binning, add_uncert=True)

    if out is None:
      out = [copy.deepcopy(tmp_hist), copy.deepcopy(tmp_hist_uncert), copy.deepcopy(bins)]
    else:
      out[0] += tmp_hist
      out[1] = np.sqrt(out[1]**2 + tmp_hist_uncert**2)
    return out


  def _method_histogram_2d(self, tmp, out, column, bins=5):
    if self.wt_name is None:
      tmp_hist, binsx, binsy = np.histogram2d(tmp[column[0]], tmp[column[1]], bins=bins)
    else:
      tmp_hist, binsx, binsy = np.histogram2d(tmp[column[0]], tmp[column[1]], weights=tmp[self.wt_name], bins=bins)

    if out is None:
      out = [copy.deepcopy(tmp_hist), copy.deepcopy((binsx,binsy))]
    else:
      out[0] += tmp_hist
    return out


  def _method_histogram_2d_and_uncert(self, tmp, out, column, bins=5):
    if self.wt_name is None:
      tmp_hist, binsx, binsy = np.histogram2d(tmp[column[0]], tmp[column[1]], bins=bins)
      tmp_hist_uncert = np.sqrt(tmp_hist)
    else:
      tmp_hist, binsx, binsy = np.histogram2d(tmp[column[0]], tmp[column[1]], weights=tmp[self.wt_name], bins=bins)
      tmp_hist_wt_squared, _, _ = np.histogram2d(tmp[column[0]], tmp[column[1]], weights=tmp[self.wt_name]**2, bins=bins)
      tmp_hist_uncert = np.sqrt(tmp_hist_wt_squared)

    if out is None:
      out = [copy.deepcopy(tmp_hist), copy.deepcopy(tmp_hist_uncert), copy.deepcopy((binsx,binsy))]
    else:
      out[0] += tmp_hist
      out[1] = np.sqrt(out[1]**2 + tmp_hist_uncert**2)
    return out


  def _method_min_max(self, tmp, out):

    tmp_min_max = {col: [np.min(tmp[col]), np.max(tmp[col])] for col in tmp.columns}

    if out is None:
      out = copy.deepcopy(tmp_min_max)
    else:
      out = {col : [min(out[col][0], tmp_min_max[col][0]), max(out[col][1], tmp_min_max[col][1])] for col in tmp.columns}
    return out


  def _method_sampled_dataset(self, tmp, out, sampling_fraction=1.0):

    if sampling_fraction != 1.0:
      indices = np.random.choice(tmp.index, int(len(tmp)*sampling_fraction), replace=False)
      tmp = tmp.loc[indices,:]

    if out is None:
      out = copy.deepcopy(tmp)
    else:
      out = pd.concat([out, tmp], axis=0, ignore_index=True)
    return out


  def _method_sum(self, tmp, out, count=False):
    if self.wt_name is None or count:
      tmp_total = len(tmp)
    else:
      tmp_total = float(np.sum(tmp[self.wt_name]))

    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      out += tmp_total
    return out


  def _method_sum_formula(self, tmp, out, column):
    tmp["tmp_for_sum_formula"] = tmp.eval(column)
    tmp_total = float(np.sum(tmp["tmp_for_sum_formula"]))
    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      out += tmp_total
    return out


  def _method_sum_w_unique_columns(self, tmp, out, unique_combinations, count=False):

    for ind, uc in unique_combinations.iterrows():

      selection = " & ".join([f"({column}=={float(unique_combinations.loc[ind,column])})" for column in list(unique_combinations.columns)])
      selection = self._ReplaceEqualsWithIsClose(selection)
      tmp_tmp = tmp.loc[tmp.eval(selection),:]

      if self.wt_name is None or count:
        tmp_total = len(tmp_tmp)
      else:
        tmp_total = float(np.sum(tmp_tmp[self.wt_name]))

      if out is None:
        out = copy.deepcopy(unique_combinations)
        out["sum_w"] = 0.0

      out.loc[ind, "sum_w"] = float(out.loc[ind, "sum_w"]) + tmp_total

    return out


  def _method_sum_w2(self, tmp, out):

    if self.wt_name is None:
      tmp_total = len(tmp)
    else:
      tmp_total = float(np.sum(tmp[self.wt_name]**2))

    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      out += tmp_total
    return out


  def _method_sum_w2_unique_columns(self, tmp, out, unique_combinations):

    for ind, uc in unique_combinations.iterrows():

      selection = " & ".join([f"({column}=={float(unique_combinations.loc[ind,column])})" for column in list(unique_combinations.columns)])
      selection = self._ReplaceEqualsWithIsClose(selection)
      tmp_tmp = tmp.loc[tmp.eval(selection),:]

      if self.wt_name is None:
        tmp_total = len(tmp_tmp)
      else:
        tmp_total = float(np.sum(tmp_tmp[self.wt_name]**2))

      if out is None:
        out = copy.deepcopy(unique_combinations)
        out["sum_w2"] = 0.0

      out.loc[ind, "sum_w2"] = float(out.loc[ind, "sum_w2"]) + tmp_total

    return out


  def _method_sum_columns(self, tmp, out):
    if self.wt_name is None:
      tmp_total_column = {col : float(np.sum(tmp[col])) for col in tmp.columns}
      tmp_total_wt = len(tmp)
    else:
      tmp_total_column = {col : float(np.sum(tmp[self.wt_name]*tmp[col])) for col in tmp.columns if col != self.wt_name}
      tmp_total_wt = float(np.sum(tmp[self.wt_name]))

    if out is None:
      out = [copy.deepcopy(tmp_total_column), copy.deepcopy(tmp_total_wt)]
    else:
      out = [
        {col : val + out[0][col] for col, val in tmp_total_column.items()},
        out[1] + tmp_total_wt
      ]
    return out


  def _method_part_std(self, tmp, out, means):

    if self.wt_name is None:
      tmp_total_column = {col : float(np.sum((tmp[col] - means[col])**2)) for col in tmp.columns}
      tmp_total_wt = len(tmp)
    else:
      tmp_total_column = {col : float(np.sum(tmp[self.wt_name]*((tmp[col] - means[col])**2))) for col in tmp.columns if col != self.wt_name}
      tmp_total_wt = float(np.sum(tmp[self.wt_name]))

    if out is None:
      out = [copy.deepcopy(tmp_total_column), copy.deepcopy(tmp_total_wt)]
    else:
      out = [
        {col : val + out[0][col] for col, val in tmp_total_column.items()},
        out[1] + tmp_total_wt
      ]
    return out


  def _method_train_test_split(self, tmp, out, column, test_fraction=0.2):

    train, test = train_test_split(tmp, test_size=test_fraction, random_state=42)
    if out is None:
      out = [copy.deepcopy(train), copy.deepcopy(test)]
    else:
      out[0] = pd.concat([out[0], train], axis=0, ignore_index=True)
      out[1] = pd.concat([out[1], test], axis=0, ignore_index=True)
    return out


  def _method_unique(self, tmp, out, unique_threshold=20):
    tmp_unique = {}
    for col in tmp.columns:
      if col == self.wt_name: continue
      unique = [float(i) for i in np.unique(tmp[col])]
      if unique is None:
        tmp_unique[col] = None
      else:
        if len(unique) < unique_threshold:
          tmp_unique[col] = copy.deepcopy(unique)
        else:
          tmp_unique[col] = None

    if out is None:
      out = copy.deepcopy(tmp_unique)
    else:
      for k, v in tmp_unique.items():
        if v is None or out[k] is None:
          out[k] = None
        else:
          sets = sorted(list(set(out[k] + v)))
          if len(sets) < unique_threshold:
            out[k] = copy.deepcopy(sets)
          else:
            out[k] = None
    return out

  def _method_part_quantile(self, tmp, out, column, quantile):

    if quantile not in [0,1]:

      # Find quantile of batch
      tmp = tmp.reset_index(drop=True)
      sorter = np.argsort(tmp[column].to_numpy())
      tmp = tmp.loc[sorter,:]
      tmp_wt_name = False
      if self.wt_name is None:
        tmp_wt_name = True
        self.wt_name = "wt"
        tmp[self.wt_name] = np.array(range(1,len(tmp)+1))
      total = float(np.sum(tmp[self.wt_name]))
      cum_sum = np.cumsum(tmp[self.wt_name])/float(np.sum(tmp[self.wt_name]))
      if len(cum_sum) == 0: 
        return out
      closest_index = (cum_sum - quantile).abs().idxmin()
      interval = float(tmp.loc[closest_index, column])
      if tmp_wt_name:
        self.wt_name = None

      # do some averaging
      if out is None:
        out = [interval, total]
      else:
        out = [
          (out[0]*out[1] + interval*total)/ (out[1]+total),
          out[1]+total
        ]
  
    elif quantile == 0:

      min_out = min(tmp[column].to_numpy().flatten())

      if out is None:
        out = [min_out]
      else:
        out = [min(min_out,out[0])]

    elif quantile == 1:

      max_out = max(tmp[column].to_numpy().flatten())

      if out is None:
        out = [max_out]
      else:
        out = [max(max_out,out[0])]

    return out
