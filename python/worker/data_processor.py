import copy
import pandas as pd
import numpy as np

from data_loader import DataLoader
from useful_functions import CustomHistogram

class DataProcessor():

  def __init__(
      self, 
      datasets, 
      dataset_type, 
      wt_name = None, 
      n_events = None, 
      options = {}
    ):

    # Check inputs are correct
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
    self.batch_size = 10**5
    self.selection = None
    self.columns = None
    self.scale = None

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

    # Variables needed for running
    self.file_ind = 0
    self.batch_ind = 0
    self.finished = False

  def _SetOptions(
      self, 
      options
    ):

    for key, value in options.items():
      setattr(self, key, value)

  def Restart(
      self
    ):
    self.file_ind = 0
    self.batch_ind = 0
    self.finished = False

  def LoadNextBatch(
      self, 
      extra_sel = None, 
      functions_to_apply = []
    ):

    self.finished = False

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
      if column_ind == 0:
        df = copy.deepcopy(tmp)
      else:
        df = pd.concat([df, tmp], axis=1)
      del tmp

    # Apply functions
    for f in functions_to_apply:
      if isinstance(f, str):
        if f == "untransform":
          df = self.UnTransformData(df)
          df = self.ApplySelection(df, extra_sel=extra_sel)
        elif f == "transform":
          df = self.ApplySelection(df, extra_sel=extra_sel)
          df = self.TransformData(df)
      else:
        df = f(df)
    if "transform" not in functions_to_apply and "untransform" not in functions_to_apply:
      df = self.ApplySelection(df, extra_sel=extra_sel)

    # Select the columns
    if self.columns is not None:
      df = df.loc[:,[col for col in self.columns if col in df.columns]]

    # Scale weights
    if self.scale is not None:
      if self.wt_name is None: self.wt_name = "wt"
      scale = self.scale if self.dataset_type != "generator" else self.scale/self.n_events  
      if self.wt_name in df.columns:
        df.loc[:, self.wt_name] *= scale
      else:
        df.loc[:, self.wt_name] = scale
       
    # Sort columns
    df = df.loc[:, sorted(list(df.columns))]
      
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

    return df

  def GetFull(
      self, 
      method = "dataset", 
      column = None, 
      bins = 10, 
      density = False, 
      unique_threshold = 40, 
      extra_sel = None,
      functions_to_apply = [],
      discrete_binning = True,
    ):

    if method == "std": # Get the mean for standard deviation calculation
      sum_cols, sum_wts = self.GetFull(method="sum_columns")
      means = {k : v/sum_wts for k, v in sum_cols.items()}

    if method == "n_eff": # Get the number of effective events
      sum_wts_squared = self.GetFull(method="sum_w2")
      sum_wts = self.GetFull(method="sum")
      return (sum_wts**2)/sum_wts_squared

    self.batch_ind = 0
    self.file_ind = 0

    first_loop = True
    while not self.finished:

      # Load batch
      tmp = self.LoadNextBatch(
        extra_sel = extra_sel,
        functions_to_apply = functions_to_apply,
      )

      # Run method
      if method == "dataset": # get full dataset

        if first_loop:
          out = copy.deepcopy(tmp)
          first_loop = False
        else:
          out = pd.concat([out, tmp], axis=0, ignore_index=True)

      elif method == "histogram": # make a histogram from the dataset

        if self.wt_name is None:
          tmp_hist, bins = CustomHistogram(tmp.loc[:,column], bins=bins, discrete_binning=discrete_binning)
        else:
          tmp_hist, bins = CustomHistogram(tmp.loc[:,column], weights=tmp.loc[:,self.wt_name], bins=bins, discrete_binning=discrete_binning)

        if first_loop:
          out = [copy.deepcopy(tmp_hist), copy.deepcopy(bins)]
          first_loop = False
        else:
          out[0] += tmp_hist

      elif method == "histogram_and_uncert": # make a histogram from the dataset

        if self.wt_name is None:
          tmp_hist, tmp_hist_uncert, bins = CustomHistogram(tmp.loc[:,column], bins=bins, discrete_binning=discrete_binning, add_uncert=True)
        else:
          tmp_hist, tmp_hist_uncert, bins = CustomHistogram(tmp.loc[:,column], weights=tmp.loc[:,self.wt_name], bins=bins, discrete_binning=discrete_binning, add_uncert=True)

        if first_loop:
          out = [copy.deepcopy(tmp_hist), copy.deepcopy(tmp_hist_uncert), copy.deepcopy(bins)]
          first_loop = False
        else:
          out[0] += tmp_hist
          out[1] = np.sqrt(out[1]**2 + tmp_hist_uncert**2)

      elif method in ["sum","count"]: # sum up the weights or count events

        if self.wt_name is None or method == "count":
          tmp_total = len(tmp)
        else:
          tmp_total = float(np.sum(tmp.loc[:,self.wt_name]))

        if first_loop:
          out = copy.deepcopy(tmp_total)
          first_loop = False
        else:
          out += tmp_total

      elif method == "sum_w2": # sum up the weights or count events

        if self.wt_name is None:
          tmp_total = len(tmp)
        else:
          tmp_total = float(np.sum(tmp.loc[:,self.wt_name]**2))

        if first_loop:
          out = copy.deepcopy(tmp_total)
          first_loop = False
        else:
          out += tmp_total

      elif method in ["sum_columns","mean"]: # find sum of columns - also used for the mean

        if self.wt_name is None:
          tmp_total_column = {col : float(np.sum(tmp.loc[:,col])) for col in tmp.columns}
          tmp_total_wt = len(tmp)
        else:
          tmp_total_column = {col : float(np.sum(tmp.loc[:,self.wt_name]*tmp.loc[:,col])) for col in tmp.columns if col != self.wt_name}
          tmp_total_wt = float(np.sum(tmp.loc[:,self.wt_name]))

        if first_loop:
          out = [copy.deepcopy(tmp_total_column), copy.deepcopy(tmp_total_wt)]
          first_loop = False
        else:
          out = [
            {col : val + out[0][col] for col, val in tmp_total_column.items()},
            out[1] + tmp_total_wt
          ]

      elif method == "std": # find std of columns

        if self.wt_name is None:
          tmp_total_column = {col : float(np.sum((tmp.loc[:,col] - means[col])**2)) for col in tmp.columns}
          tmp_total_wt = len(tmp)
        else:
          tmp_total_column = {col : float(np.sum(tmp.loc[:,self.wt_name]*((tmp.loc[:,col] - means[col])**2))) for col in tmp.columns if col != self.wt_name}
          tmp_total_wt = float(np.sum(tmp.loc[:,self.wt_name]))

        if first_loop:
          out = [copy.deepcopy(tmp_total_column), copy.deepcopy(tmp_total_wt)]
          first_loop = False
        else:
          out = [
            {col : val + out[0][col] for col, val in tmp_total_column.items()},
            out[1] + tmp_total_wt
          ]

      elif method == "unique": # find unique values of a column

        tmp_unique = {}
        for col in tmp.columns:
          if col == self.wt_name: continue
          unique = list(np.unique(tmp.loc[:, col]))
          if unique is None:
            tmp_unique[col] = None
          else:
            if len(unique) < unique_threshold:
              tmp_unique[col] = copy.deepcopy(unique)
            else:
              tmp_unique[col] = None

        if first_loop:
          out = copy.deepcopy(tmp_unique)
          first_loop = False
        else:
          for k, v in tmp_unique.items():
            if v is None or out[k] is None:
              out[k] = None
            else:
              sets = list(set(out[k] + v))
              if len(sets) < unique_threshold:
                out[k] = copy.deepcopy(sets)
              else:
                out[k] = None

          # Remove tmp from memory
          del tmp

      else: 
        out = None

    self.finished = False

    if method == "mean": # calculate means
      return {k : v/out[1] for k, v in out[0].items()}

    if method == "std": # calculate std
      return {k : float(np.sqrt(v/out[1])) for k, v in out[0].items()}

    if method == "histogram" and density: # normalise histograms
      sum_wt = np.sum(out[0])
      return out[0]/sum_wt, out[1]

    if method == "histogram_and_uncert" and density: # normalise histograms
      sum_wt = np.sum(out[0])
      return out[0]/sum_wt, out[1]/sum_wt, out[2]

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
      if column_name in self.parameters["standardisation"]:
          data.loc[:,column_name] = self.Standardise(data.loc[:,column_name], column_name)
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
      if column_name in self.parameters["standardisation"]:
        data.loc[:,column_name] = self.UnStandardise(data.loc[:,column_name], column_name)
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
      data = data.loc[data.eval(selection),:]
    return data