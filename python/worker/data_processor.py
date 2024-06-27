import copy
import pandas as pd
import numpy as np

from data_loader import DataLoader
from useful_functions import CustomHistogram, Resample

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
    self.resample = False
    self.resampling_seed = 1
    self.functions = []

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

    functions_to_apply = self.functions + functions_to_apply

    self.finished = False

    for column_ind in range(len(self.datasets[self.file_ind])):

      # Get dataset
      if self.dataset_type == "dataset":
        tmp = self.datasets[self.file_ind][column_ind]
      elif self.dataset_type == "generator":
        tmp = self.datasets[self.file_ind][column_ind](self.n_events_per_batch[self.file_ind][self.batch_ind]) # Need to evolve the seed, should be able to set seeds in list and sample through them each time
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
       
    # Resample
    if self.resample and self.wt_name is not None:
      columns_without_weights = list(df.columns)
      columns_without_weights.remove(self.wt_name)
      data, wts = Resample(df.loc[:,columns_without_weights].to_numpy(), df.loc[:,self.wt_name].to_numpy(), n_samples=int(np.floor(np.sum(df.loc[:,self.wt_name].to_numpy()))), seed=self.resampling_seed)
      df = pd.DataFrame(np.hstack((data,wts.reshape(-1,1))), columns=columns_without_weights+[self.wt_name], dtype=np.float64)

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
      extra_sel = None,
      functions_to_apply = [],
      bins = 10, 
      density = False, 
      unique_threshold = 40, 
      discrete_binning = True,
      quantile = 0.01,
      ignore_quantile = 0.005,
      custom = None,
      custom_options = {},
    ):

    if method == "std": # Get the mean for standard deviation calculation
      sum_cols, sum_wts = self.GetFull(method="sum_columns", extra_sel=extra_sel, functions_to_apply=functions_to_apply)
      means = {k : v/sum_wts for k, v in sum_cols.items()}
    elif method == "n_eff": # Get the number of effective events
      sum_wts_squared = self.GetFull(method="sum_w2", extra_sel=extra_sel, functions_to_apply=functions_to_apply)
      sum_wts = self.GetFull(method="sum", extra_sel=extra_sel, functions_to_apply=functions_to_apply)
      return (sum_wts**2)/sum_wts_squared
    elif method == "bins_with_equal_spacing": # Get equally spaced bins
      return list(np.linspace(self.GetFull(method="quantile", extra_sel=extra_sel, functions_to_apply=functions_to_apply, column=column, quantile=ignore_quantile), self.GetFull(method="quantile", extra_sel=extra_sel, functions_to_apply=functions_to_apply, column=column, quantile=1-ignore_quantile), num=bins+1))
    elif method == "bins_with_equal_stats": # Get equal stat bins
      return [self.GetFull(method="quantile", extra_sel=extra_sel, functions_to_apply=functions_to_apply, column=column, quantile=i) for i in np.linspace(ignore_quantile, 1-ignore_quantile, num=bins+1)]

    self.batch_ind = 0
    self.file_ind = 0
    out = None

    while not self.finished:

      # Load batch
      tmp = self.LoadNextBatch(
        extra_sel = extra_sel,
        functions_to_apply = functions_to_apply,
      )

      # Run method
      if method in ["dataset"]: # get full dataset
        out = self._method_dataset(tmp, out)
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
      elif method in ["count"]: # count events
        out = self._method_sum(tmp, out, count=True)
      elif method in ["sum_w2"]: # sum up the weights or count events
        out = self._method_sum_w2(tmp, out)
      elif method in ["sum_columns","mean"]: # find sum of columns - also used for the mean
        out = self._method_sum_columns(tmp, out)
      elif method in ["std"]: # find partial information for std of columns
        out = self._method_part_std(tmp, out, means)
      elif method in ["unique"]: # find unique values of a column
        out = self._method_unique(tmp, out, unique_threshold)
      elif method in ["quantile"]: # find a quantile of the dataset
        out = self._method_part_quantile(tmp, out, column, quantile)
      elif method in ["custom"]: # custom function
        out = custom(tmp, out, options=custom_options)
      else: 
        out = None

      # Remove tmp from memory
      del tmp

    self.finished = False

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
      if column_name == "prob":
        for col in self.parameters["X_columns"]:
          data.loc[:,column_name] /= self.parameters["standardisation"][col]["std"]
      elif column_name == "log_prob":
        for col in self.parameters["X_columns"]:
          data.loc[:,column_name] -= np.log(self.parameters["standardisation"][col]["std"])
      elif column_name in list(self.parameters["standardisation"].keys()):
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

  def _method_dataset(self, tmp, out):
    if out is None:
      out = copy.deepcopy(tmp)
    else:
      out = pd.concat([out, tmp], axis=0, ignore_index=True)
    return out

  def _method_histogram(self, tmp, out, column, bins=40, discrete_binning=False):
    if self.wt_name is None:
      tmp_hist, bins = CustomHistogram(tmp.loc[:,column], bins=bins, discrete_binning=discrete_binning)
    else:
      tmp_hist, bins = CustomHistogram(tmp.loc[:,column], weights=tmp.loc[:,self.wt_name], bins=bins, discrete_binning=discrete_binning)

    if out is None:
      out = [copy.deepcopy(tmp_hist), copy.deepcopy(bins)]
    else:
      out[0] += tmp_hist
    return out

  def _method_histogram_and_uncert(self, tmp, out, column, bins=40, discrete_binning=False):
    if self.wt_name is None:
      tmp_hist, tmp_hist_uncert, bins = CustomHistogram(tmp.loc[:,column], bins=bins, discrete_binning=discrete_binning, add_uncert=True)
    else:
      tmp_hist, tmp_hist_uncert, bins = CustomHistogram(tmp.loc[:,column], weights=tmp.loc[:,self.wt_name], bins=bins, discrete_binning=discrete_binning, add_uncert=True)

    if out is None:
      out = [copy.deepcopy(tmp_hist), copy.deepcopy(tmp_hist_uncert), copy.deepcopy(bins)]
    else:
      out[0] += tmp_hist
      out[1] = np.sqrt(out[1]**2 + tmp_hist_uncert**2)
    return out

  def _method_histogram_2d(self, tmp, out, column, bins=5):
    if self.wt_name is None:
      tmp_hist, binsx, binsy = np.histogram2d(tmp.loc[:,column[0]], tmp.loc[:,column[1]], bins=bins)
    else:
      tmp_hist, binsx, binsy = np.histogram2d(tmp.loc[:,column[0]], tmp.loc[:,column[1]], weights=tmp.loc[:,self.wt_name], bins=bins)

    if out is None:
      out = [copy.deepcopy(tmp_hist), copy.deepcopy((binsx,binsy))]
    else:
      out[0] += tmp_hist
    return out

  def _method_histogram_2d_and_uncert(self, tmp, out, column, bins=5):
    if self.wt_name is None:
      tmp_hist, binsx, binsy = np.histogram2d(tmp.loc[:,column[0]], tmp.loc[:,column[1]], bins=bins)
      tmp_hist_uncert = np.sqrt(tmp_hist)
    else:
      tmp_hist, binsx, binsy = np.histogram2d(tmp.loc[:,column[0]], tmp.loc[:,column[1]], weights=tmp.loc[:,self.wt_name], bins=bins)
      tmp_hist_wt_squared, _, _ = np.histogram2d(tmp.loc[:,column[0]], tmp.loc[:,column[1]], weights=tmp.loc[:,self.wt_name]**2, bins=bins)
      tmp_hist_uncert = np.sqrt(tmp_hist_wt_squared)

    if out is None:
      out = [copy.deepcopy(tmp_hist), copy.deepcopy(tmp_hist_uncert), copy.deepcopy((binsx,binsy))]
    else:
      out[0] += tmp_hist
      out[1] = np.sqrt(out[1]**2 + tmp_hist_uncert**2)
    return out

  def _method_sum(self, tmp, out, count=False):
    if self.wt_name is None or count:
      tmp_total = len(tmp)
    else:
      tmp_total = float(np.sum(tmp.loc[:,self.wt_name]))

    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      out += tmp_total
    return out

  def _method_sum_w2(self, tmp, out):

    if self.wt_name is None:
      tmp_total = len(tmp)
    else:
      tmp_total = float(np.sum(tmp.loc[:,self.wt_name]**2))

    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      out += tmp_total
    return out

  def _method_sum_columns(self, tmp, out):
    if self.wt_name is None:
      tmp_total_column = {col : float(np.sum(tmp.loc[:,col])) for col in tmp.columns}
      tmp_total_wt = len(tmp)
    else:
      tmp_total_column = {col : float(np.sum(tmp.loc[:,self.wt_name]*tmp.loc[:,col])) for col in tmp.columns if col != self.wt_name}
      tmp_total_wt = float(np.sum(tmp.loc[:,self.wt_name]))

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
      tmp_total_column = {col : float(np.sum((tmp.loc[:,col] - means[col])**2)) for col in tmp.columns}
      tmp_total_wt = len(tmp)
    else:
      tmp_total_column = {col : float(np.sum(tmp.loc[:,self.wt_name]*((tmp.loc[:,col] - means[col])**2))) for col in tmp.columns if col != self.wt_name}
      tmp_total_wt = float(np.sum(tmp.loc[:,self.wt_name]))

    if out is None:
      out = [copy.deepcopy(tmp_total_column), copy.deepcopy(tmp_total_wt)]
    else:
      out = [
        {col : val + out[0][col] for col, val in tmp_total_column.items()},
        out[1] + tmp_total_wt
      ]
    return out

  def _method_unique(self, tmp, out, unique_threshold=20):
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

    if out is None:
      out = copy.deepcopy(tmp_unique)
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
    return out

  def _method_part_quantile(self, tmp, out, column, quantile):

    # Find quantile of batch
    tmp = tmp.reset_index(drop=True)
    sorter = np.argsort(tmp.loc[:,column].to_numpy())
    tmp = tmp.loc[sorter,:]
    tmp_wt_name = False
    if self.wt_name is None:
      tmp_wt_name = True
      self.wt_name = "wt"
      tmp.loc[:,self.wt_name] = np.array(range(1,len(tmp)+1))
    total = float(np.sum(tmp.loc[:,self.wt_name]))
    cum_sum = np.cumsum(tmp.loc[:,self.wt_name])/float(np.sum(tmp.loc[:,self.wt_name]))
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
    return out