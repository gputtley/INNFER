import gc
import yaml
import copy
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from data_loader import DataLoader
from plotting import plot_histograms
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

class PreProcess():
  """
  PreProcess class for preprocessing data prior to be used for training.
  """
  def __init__(self, parquet_file_name=None, X_columns=None, Y_columns=None, options={}):
    """
    Initialize the PreProcess instance.

    Args:
        parquet_file_name (str): The name of the Parquet file containing the dataset.
        X_columns (list): List of column names to be used as features.
        Y_columns (list): List of column names to be used as target variables.
        options (dict): Additional options for customisation.
    """
    self.parquet_file_name = parquet_file_name
    self.X_columns = X_columns
    self.Y_columns = Y_columns
    self.standardise = []
    self.train_test_val_split = "0.3:0.3:0.4"
    self.equalise_y_wts = False
    self.parameters = {}
    self.output_dir = "data/"
    self.plot_dir = "plots/"
    self._SetOptions(options)

  def _SetOptions(self, options):
    """
    Set options for the PreProcess instance.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):
    """
    Execute the data preprocessing pipeline.
    """
    dl = DataLoader(self.parquet_file_name)
    full_dataset = dl.LoadFullDataset()

    Y_columns = [element for element in self.Y_columns if element in dl.columns]
    self.parameters["X_columns"] = self.X_columns
    self.parameters["Y_columns"] = Y_columns

    self.parameters["unique_Y_values"] = {}
    for col in full_dataset[Y_columns].columns:
      unique_vals = full_dataset.loc[:,[col]].drop_duplicates()
      if len(unique_vals) < 20:
        self.parameters["unique_Y_values"][col] = sorted([float(i) for i in unique_vals.to_numpy().flatten()])
      else:
        self.parameters["unique_Y_values"][col] = None

    train_ratio = float(self.train_test_val_split.split(":")[0])
    test_ratio = float(self.train_test_val_split.split(":")[1])
    val_ratio = float(self.train_test_val_split.split(":")[2])   

    train_df, temp_df = train_test_split(full_dataset, test_size=(1 - train_ratio), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    del dl, full_dataset, temp_df, unique_vals
    gc.collect()

    self.parameters["standardisation"] = {}
    for column_name in train_df.columns:
      if column_name in self.standardise:
        self.parameters["standardisation"][column_name] = self.GetStandardisationParameters(train_df.loc[:,column_name])

    train_df = self.TransformData(train_df)
    test_df = self.TransformData(test_df)
    val_df = self.TransformData(val_df)

    X_train_df = train_df[self.X_columns]
    X_test_df = test_df[self.X_columns]
    X_val_df = val_df[self.X_columns]

    Y_train_df = train_df[Y_columns]
    Y_test_df = test_df[Y_columns]
    Y_val_df = val_df[Y_columns]

    wt_train_df = train_df[["wt"]]
    wt_test_df = test_df[["wt"]]
    wt_val_df = val_df[["wt"]]      

    if self.equalise_y_wts:
      wt_train_df = self.EqualiseYWeights(Y_train_df, wt_train_df)
      wt_test_df = self.EqualiseYWeights(Y_test_df, wt_test_df)
      wt_val_df = self.EqualiseYWeights(Y_val_df, wt_val_df)

    pq.write_table(pa.Table.from_pandas(X_train_df), self.output_dir+"/X_train.parquet")
    pq.write_table(pa.Table.from_pandas(X_test_df), self.output_dir+"/X_test.parquet")
    pq.write_table(pa.Table.from_pandas(X_val_df), self.output_dir+"/X_val.parquet")
    pq.write_table(pa.Table.from_pandas(Y_train_df), self.output_dir+"/Y_train.parquet")
    pq.write_table(pa.Table.from_pandas(Y_test_df), self.output_dir+"/Y_test.parquet")
    pq.write_table(pa.Table.from_pandas(Y_val_df), self.output_dir+"/Y_val.parquet")
    pq.write_table(pa.Table.from_pandas(wt_train_df), self.output_dir+"/wt_train.parquet")
    pq.write_table(pa.Table.from_pandas(wt_test_df), self.output_dir+"/wt_test.parquet")
    pq.write_table(pa.Table.from_pandas(wt_val_df), self.output_dir+"/wt_val.parquet")

    with open(self.output_dir+"/parameters.yaml", 'w') as yaml_file:
      yaml.dump(self.parameters, yaml_file, default_flow_style=False)        

  def GetStandardisationParameters(self, column):
    """
    Get mean and standard deviation for standardisation.

    Args:
        column (pd.Series): The column for which to calculate parameters.

    Returns:
        dict: Dictionary containing mean and standard deviation.
    """
    return {"mean": float(column.mean()), "std": float(column.std()) if column.std() !=0 else 1.0}

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
  
  def EqualiseYWeights(self, Y, wt):
    """
    Equalize weights for target variables.

    Args:
        Y (pd.DataFrame): DataFrame containing target variables.
        wt (pd.DataFrame): DataFrame containing weights.

    Returns:
        pd.DataFrame: DataFrame with equalized weights.
    """
    unique_rows = Y.drop_duplicates()

    sum_weights = []
    for _, ur in unique_rows.iterrows():
      matching_rows = (Y == ur).all(axis=1)
      sum_weights.append(float(wt[matching_rows].sum().iloc[0]))
    max_sum_weights = max(sum_weights)

    ind = 0
    for _, ur in unique_rows.iterrows():
      if sum_weights[ind] == 0: continue
      matching_rows = (Y == ur).all(axis=1)
      wt.loc[matching_rows,:] = (max_sum_weights / sum_weights[ind]) * wt.loc[matching_rows,:]
      ind += 1

    return wt

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

  def UnTransformProb(self, prob):
    """
    Untransform probabilities.

    Args:
        prob (float): The probability to be untransformed.

    Returns:
        float: The untransformed probability.
    """    
    for column_name in self.parameters["Y_columns"]:
      if column_name in self.parameters["standardisation"]:
        prob *= self.parameters["standardisation"][column_name]["std"]
    return prob 

  def LoadSplitData(self, dataset="train", get=["X","Y","wt"]):
    """
    Load split data (X, Y, and weights) from the preprocessed files.

    Args:
        dataset (str): The dataset split to load ("train", "test", or "val").
        get (list): List of data components to retrieve ("X", "Y", "wt").

    Returns:
        list: List containing loaded data components based on the 'get' parameter.
    """
    output = []
    if "X" in get:
      X_dl = DataLoader(f"{self.output_dir}/X_{dataset}.parquet")
      X_data = X_dl.LoadFullDataset()
      X_data = self.UnTransformData(X_data)
      output.append(X_data)

    if "Y" in get:
      Y_dl = DataLoader(f"{self.output_dir}/Y_{dataset}.parquet")
      Y_data = Y_dl.LoadFullDataset()
      Y_data = self.UnTransformData(Y_data)
      output.append(Y_data)

    if "wt" in get:
      wt_dl = DataLoader(f"{self.output_dir}/wt_{dataset}.parquet")
      output.append(wt_dl.LoadFullDataset())

    return output

  def PlotX(self, vary, freeze={}, n_bins=40, ignore_quantile=0.01, dataset="train", extra_name=""):
    """
    Plot histograms for X variables.

    Args:
        vary (str): The variable to vary the plot against.
        freeze (dict): Dictionary specifying conditions to freeze variables.
        n_bins (int): Number of bins for histograms.
        ignore_quantile (float): Quantile value to ignore when defining bin range.
        dataset (str): The dataset split to use for plotting ("train", "test", or "val").
        extra_name (str): Extra name to append to the plot file.

    Returns:
        None
    """
    # Loads data
    X_data, Y_data, wt_data = self.LoadSplitData(dataset=dataset)

    # Sets freeze conditions
    final_condition = None
    for col, val in freeze.items():
      if col in Y_data.columns:
        if val == "central":
          uv = np.sort(np.unique(Y_data.loc[:,col], axis=0))
          val = uv[len(uv) // 2]
        col_condition = (Y_data.loc[:,col] == val)
        if final_condition is None:
            final_condition = col_condition
        else:
            final_condition = final_condition & col_condition
    if final_condition is not None:
      indices = Y_data[final_condition].index
      X_data = X_data.iloc[indices].reset_index(drop=True)
      Y_data = Y_data.iloc[indices].reset_index(drop=True)
      wt_data = wt_data.iloc[indices].reset_index(drop=True)

    # Loop through X variables
    for col in X_data.columns:

      # Find combined binning first
      trimmed_data = X_data.loc[:,[col]]
      trimmed_data.loc[:,"wt"] = wt_data
      lower_value = np.quantile(trimmed_data.loc[:,col], ignore_quantile)
      upper_value = np.quantile(trimmed_data.loc[:,col], 1-ignore_quantile)
      trimmed_data = trimmed_data[(trimmed_data.loc[:,col] >= lower_value) & (trimmed_data.loc[:,col] <= upper_value)]
      _, bins = np.histogram(trimmed_data.loc[:,col], weights=trimmed_data.loc[:,"wt"], bins=min(n_bins,len(np.unique(trimmed_data))))

      # Loop through varied variables
      hists = []
      hist_names = []
      if vary not in Y_data.columns:
        unique_rows = [None]
      else:
        unique_rows = np.unique(Y_data.loc[:,vary], axis=0)
      for ur in unique_rows:
        if ur != None:
          condition = Y_data.loc[:,vary] == ur
          matching_rows = Y_data[condition].index
          X_cut = X_data.iloc[matching_rows].reset_index(drop=True)
          wt_cut = wt_data.iloc[matching_rows].reset_index(drop=True)
          hist_names.append(f"{vary}={ur}")
        else:
          X_cut = copy.deepcopy(X_data)
          wt_cut = copy.deepcopy(wt_data)
          hist_names.append(None)
        h, _ = np.histogram(X_cut.loc[:,col], bins=bins, weights=wt_cut.loc[:,"wt"])
        hists.append(h)

      # Plot histograms
      plot_name = self.plot_dir+f"/X_distributions_varying_{vary}_against_{col}{extra_name}"
      plot_histograms(
        bins[:-1],
        hists,
        hist_names,
        title_right = "",
        name = plot_name,
        x_label = col,
        y_label = "Events",
        anchor_y_at_0 = True
      )


  def PlotY(self, n_bins=40, dataset="train"):
    """
    Plot histograms for Y variables.

    Args:
        n_bins (int): Number of bins for histograms.
        dataset (str): The dataset split to use for plotting ("train", "test", or "val").

    Returns:
        None
    """
    Y_data, wt_data = self.LoadSplitData(get=["Y","wt"], dataset=dataset)

    for col in Y_data.columns:

      hist, bins = np.histogram(Y_data.loc[:,col], bins=n_bins, weights=wt_data.loc[:,"wt"])

      plot_name = self.plot_dir+f"/Y_distributions_for_{col}"
      plot_histograms(
        bins[:-1],
        [hist],
        [None],
        title_right = "",
        name = plot_name,
        x_label = col,
        y_label = "Events",
        anchor_y_at_0 = True,
        drawstyle = "steps",
      )