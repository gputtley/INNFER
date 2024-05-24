import copy
import pandas as pd
import numpy as np

from data_loader import DataLoader
from useful_functions import CustomHistogram

class DataProcessor():

  def __init__(self, datasets, dataset_type, wt_name=None, n_events=None, options={}):

    if dataset_type not in ["dataset","parquet","generator"]:
      raise ValueError("dataset_type given is incorrect.")
    if dataset_type in ["generator"] and n_events is None:
      raise ValueError("Must specify n_events when using generators.")

    if isinstance(datasets, list):
      if isinstance(datasets[0], list):
        self.datasets = datasets
      else:
        self.datasets = [datasets]
    else:
      self.datasets = [[datasets]]

    self.dataset_type = dataset_type
    self.wt_name = wt_name
    self.n_events = n_events

    self.batch_size = 10**5
    self.selection = None
    self.columns = None

    self.parameters = {}

    self._SetOptions(options)

    if dataset_type == "dataset":
      self.batch_size = max([len(ds[0]) for ds in self.datasets])
      self.num_batches = [1 for _ in datasets]
    elif dataset_type == "generator":
      self.num_batches = [int(np.ceil(n_events/self.batch_size)) for ds in self.datasets]
      self.n_events_per_batch = [[self.batch_size]*(num_in_batch-1) + [n_events % self.batch_size] for num_in_batch in self.num_batches]
    elif dataset_type == "parquet":
      self.data_loaders = [[DataLoader(d, batch_size=self.batch_size) for d in ds] for ds in self.datasets]
      self.num_batches = [self.data_loaders[ind][0].num_batches for ind in range(len(self.datasets))]

    self.file_ind = 0
    self.batch_ind = 0

  def _SetOptions(self, options):

    for key, value in options.items():
      setattr(self, key, value)

  def LoadNextBatch(self, transform = False, untransform = False):

    for column_ind in range(len(self.datasets[self.file_ind])):

      # Get dataset
      if self.dataset_type == "dataset":
        tmp = self.datasets[self.file_ind][column_ind]
      elif self.dataset_type == "generator":
        tmp = self.datasets[self.file_ind][column_ind](self.n_events_per_batch[self.file_ind][self.batch_ind])
      elif self.dataset_type == "parquet":
        tmp = self.data_loaders[self.file_ind][column_ind].LoadNextBatch()

      # Combine columns into a full batch
      if column_ind == 0:
        df = copy.deepcopy(tmp)
      else:
        df = pd.concat([df, tmp], axis=1)
      del tmp

    # Perform actions on the dataset
    if transform:
      df = self.TransformData(df)
    if untransform:
      df = self.UnTransformData(df)
    if self.selection is not None:
      df = df.loc[col_df.eval(self.selection),:]
    if self.columns is not None:
      df = df.loc[:,self.columns]

    # Change batch and file ind
    if self.batch_ind + 1 == self.num_batches[self.file_ind]:
      self.batch_ind = 0
      if self.file_ind + 1 == len(self.datasets):
        self.file_ind += 1
      else:
        self.file_ind = 0
    else:
      self.batch_ind += 1

    return df

  def LoadFullDataset(self):

    self.batch_ind = 0
    self.file_ind = 0

    first_loop = True
    for row_ind in range(len(self.datasets)):
      for file_ind in range(len(self.datasets[row_ind])):
        for batch_ind in range(self.num_batches[self.file_ind]):

            # Load batch
            tmp = self.LoadNextBatch()

            # Combine batches into a full dataset
            if first_loop:
              df = copy.deepcopy(tmp)
              first_loop = False
            else:
              df = pd.concat([df, tmp], axis=0, ignore_index=True)
            del tmp

      return df

  def Histogram(self, column, bins=40, density=False):

    self.batch_ind = 0
    self.file_ind = 0

    first_loop = True
    for row_ind in range(len(self.datasets)):
      for file_ind in range(len(self.datasets[row_ind])):
        for batch_ind in range(self.num_batches[self.file_ind]):

          # Load batch
          self.file_ind = file_ind
          self.batch_ind = batch_ind
          tmp = self.LoadNextBatch()

          # Make histogram
          if self.wt_name is None:
            tmp_hist, bins = CustomHistogram(tmp.loc[:,column], bins=bins, density=density)
          else:
            tmp_hist, bins = CustomHistogram(tmp.loc[:,column], weights=tmp.loc[:,self.wt_name], bins=bins, density=density)

          del tmp

          # Combine batches into a full dataset
          if first_loop:
            hist = copy.deepcopy(tmp_hist)
            bins = copy.deepcopy(bins)
            first_loop = False
          else:
            hist += tmp_hist

      return hist, bins

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

  def UnTransformData(self, data):
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

  def Standardise(self, column, column_name):
    """
    Standardise a column.

    Args:
        column (pd.Series): The column to be standardised.
        column_name (str): The name of the column.

    Returns:
        pd.Series: The standardised column.
    """
    return (column - self.parameters["standardisation"][column_name]["mean"])/self.parameters["standardisation"][column_name]["std"]
  
  def UnStandardise(self, column, column_name):
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