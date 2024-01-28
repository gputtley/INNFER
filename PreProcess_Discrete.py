###other_function.py, validation.py files may need to update for the validationgeneration
###model training and preprocess steps work so far.

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
from other_functions import GetYName
from scipy.interpolate import CubicSpline

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
    self.discrete_thresholds = []
    self.train_test_val_split = "0.3:0.3:0.4"
    self.remove_quantile = 0.0
    self.equalise_y_wts = False
    self.train_test_y_vals = {}
    self.validation_y_vals = {}
    self.cut_values = {}
    self.parameters = {}
    self.output_dir = "data/"
    self.plot_dir = "plots/"
    self._SetOptions(options)


    #### DEFINE the function for the threshold
  def FindDiscreteToContinuousThresholds(self, column):
      """
      Calculate the thresholds for converting discrete values to continuous ranges.
      
      Args:
          column (pd.Series): A column of discrete values.

      Returns:
          threshold (dict): A dictionary containing the threshold for each values
      """
      unique_val = sorted(column.unique()) # no needed
      thresholds = {
          0: [0.0, 1.03003003003003],
          1: [1.03003003003003, 2.186186186186186],
          2: [2.186186186186186, 2.942942942942943],
          3: [2.942942942942943, 3.8818818818818817],
          4: [3.8818818818818817, 4.694694694694695],
          5: [4.694694694694695, 5.451451451451452],
          6: [5.451451451451452, 6.1941941941941945],
          7: [6.1941941941941945, 7.0]
      }
      return thresholds

  ### Determine the spline function for the discrete function
  def Spline(self, column_name):
    """
    Create a spline function based on the unique values of a specified column.

    Args:
        column_name (str): The name of the column to create the spline for.

    Returns:
        function: A spline function.
    """
    if column_name not in self.parameters["discrete_thresholds"]:
      raise ValueError(f"Column {column_name} is not discrete.")
    
    # extract x, y points for spline
    x = np.array(list(self.parameters["discrete_thresholds"][column_name].keys()))
    y = np.array(self.parameters["unique_Y_values"][column_name])
    
    # create the function
    cs = CubicSpline(x, y)
   
    return cs

  ### Define function for data transformation between discrete and continous 
  def DiscreteToContinuous(self, data, column_name, inverse):
    """
    Apply discrete to continuous transformation on a column or its inverse.

    Args:
        data (pd.Series): The column to be transformed.
        column_name (str): The name of the column.
        inverse (bool): Whether to apply the inverse transformation.

    Returns:
        pd.Series: The transformed column.
    """
    output_ranges = self.parameters["discrete_thresholds"].get(column_name, {})
    data = copy.deepcopy(data)

    if not inverse:
      data = data.astype(float)
      spline = self.Spline(column_name)
      
      for k, v in output_ranges.items():
        if not inverse:
          # get indices
          indices = (data == k)
          n_samples = len(data[indices])
          # compute the CDF
          param_values = np.linspace(v[0], v[1], 100000)
          cdf_vals = np.cumsum(np.abs(spline(param_values))) / np.sum(np.abs(spline(param_values)))

          # normalise the CDF
          cdf_vals /= cdf_vals[-1]

          # generate random numbers
          random_nums = np.random.rand(n_samples)

          # inverse transform sampling
          data[indices] = np.interp(random_nums, cdf_vals, param_values)

        else:
          # get indices
          indices = ((data >= v[0])) & (data < v[1])

          # do inverse
          data[inverse] = k

    if inverse:
      data = data.astype(int)

    return data

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
    print(">> Loading full dataset.")
    dl = DataLoader(self.parquet_file_name)
    full_dataset1 = dl.LoadFullDataset()

    # Get X and Y column names
    print(">> Getting columns from dataset.")
    Y_columns = [element for element in self.Y_columns if element in dl.columns]
    self.parameters["X_columns"] = self.X_columns
    self.parameters["Y_columns"] = Y_columns

    # # Get unique values for validation
    # print(">> Finding unique values for validation.")
    # self.parameters["unique_Y_values"] = {}
    # for col_n in full_dataset1[Y_columns].columns:
    #   print(col_n not in self.validation_y_vals)
    #   if col_n not in self.validation_y_vals:
    #     unique_vals = full_dataset1.loc[:,[col_n]].drop_duplicates()
    #     print(len(unique_vals))
    #     if len(unique_vals) < 20:
    #       self.parameters["unique_Y_values"][col_n] = sorted([float(i) for i in unique_vals.to_numpy().flatten()])
    #     else:
    #       self.parameters["unique_Y_values"][col_n] = None
    #   else:
    #     self.parameters["unique_Y_values"][col_n] = self.validation_y_vals[col_n]
    print(">> Finding unique values for validation.")
    self.parameters["unique_Y_values"] = {}
    for col in full_dataset1.columns:  # Use full_dataset.columns if you want to check all columns
        unique_vals = full_dataset1[col].drop_duplicates()  
        if len(unique_vals) < 20:
          self.parameters["unique_Y_values"][col] = sorted(unique_vals.tolist())
        else:
          self.parameters["unique_Y_values"][col] = None
    
    print(">>>>>> Identifying discrete columns and finding thresholds.")
    self.parameters["discrete_thresholds"] = {}
    for col in self.X_columns:  # Assuming X_columns contains the feature columns you want to process
        if self.parameters["unique_Y_values"].get(col) is not None:
            # Get the unique values for the column
            column_data = full_dataset1[col]
            # Find the thresholds for the current column
            thresholds = self.FindDiscreteToContinuousThresholds(column_data)
            self.parameters["discrete_thresholds"][col] = thresholds
            print(f"Thresholds for {col}: {self.parameters['discrete_thresholds'][col]}")

    # Get sum of weights for each unique y combination
    print(">> Finding sum of weights for each unique y combination.")
    self.parameters["yield"] = {}
    if len(Y_columns) > 0:
      unique_rows = full_dataset1.loc[:,Y_columns].drop_duplicates()
      for _, ur in unique_rows.iterrows():
        matching_rows = (full_dataset1.loc[:,Y_columns] == ur).all(axis=1)
        name = GetYName(ur, purpose="file")
        self.parameters["yield"][name] = float(np.sum(full_dataset1.loc[:,"wt"][matching_rows]))
    else:
      self.parameters["yield"]["all"] = float(np.sum(full_dataset1.loc[:,"wt"]))

    # Train test split
    print(">> Train/Test/Val splitting the data.")
    train_ratio = float(self.train_test_val_split.split(":")[0])
    test_ratio = float(self.train_test_val_split.split(":")[1])
    val_ratio = float(self.train_test_val_split.split(":")[2])   
    train_df, temp_df = train_test_split(full_dataset1, test_size=(1 - train_ratio), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Rescale train test val up the full weight
    train_df.loc[:,"wt"] /= float(self.train_test_val_split.split(":")[0])
    test_df.loc[:,"wt"] /= float(self.train_test_val_split.split(":")[1])
    val_df.loc[:,"wt"] /= float(self.train_test_val_split.split(":")[2])

    # Remove some Y combinations from train, test and val datasets
    print(">> Removing some Y combinations.")
    for k, v in self.train_test_y_vals.items():
      if k in train_df.columns:
        train_df = train_df[train_df[k].isin(v)]
        test_df = test_df[test_df[k].isin(v)]
    for k, v in self.validation_y_vals.items():
      if k in val_df.columns:
        val_df = val_df[val_df[k].isin(v)]
 
    del dl, full_dataset1, temp_df
    gc.collect()

    # Get standardisation parameters
    print(">> Getting standardisation parameters.")
    self.parameters["standardisation"] = {}
    for column_name in train_df.columns:
      if column_name in self.standardise:
        self.parameters["standardisation"][column_name] = self.GetStandardisationParameters(train_df.loc[:,column_name])

    # #### CALL the threshold function and save the thresholds combined with the getting unique values for validation
    # print(">>> Indentifying discrete columns and finding threshold.")
    # self.parameters["discrete_thresholds"] = {}
    # print(self.parameters["unique_Y_values"])
    # for col in self.X_columns:
    #   if col in self.parameters["unique_Y_values"]:
    #     col_data = full_dataset[col]
    #     thresholds = self.FindDiscreteToContinuousThresholds(column_data)
    #     self.parameters["discrete_thresholds"][col] = thresholds
    #     print(self.parameters["discrete_thresholds"][col])
    # Identify discrete columns and find thresholds

    # print(">>> Identifying discrete columns and finding thresholds.")
    # self.parameters["discrete_thresholds"] = {}
    # for col in self.X_columns:  
    #   print(self.parameters["unique_Y_values"].get(col))
    #   if self.parameters["unique_Y_values"].get(col) is not None:
    #     # Get the unique values for the column
    #     column_data = full_dataset1[col]

    #     # Find the thresholds for the current column
    #     thresholds = self.FindDiscreteToContinuousThresholds(column_data)
    #     self.parameters["discrete_thresholds"][col] = thresholds
    #     print(f"Thresholds for {col}: {self.parameters['discrete_thresholds'][col]}")

    # Transform data
    print(">> Transforming data.")
    train_df = self.TransformData(train_df)
    test_df = self.TransformData(test_df)
    val_df = self.TransformData(val_df)

    # Split in X, Y and wt
    print(">> Splitting X, Y and wt datasets.")
    X_train_df = train_df[self.X_columns]
    X_test_df = test_df[self.X_columns]
    X_val_df = val_df[self.X_columns]
    Y_train_df = train_df[Y_columns]
    Y_test_df = test_df[Y_columns]
    Y_val_df = val_df[Y_columns]
    wt_train_df = train_df[["wt"]]
    wt_test_df = test_df[["wt"]]
    wt_val_df = val_df[["wt"]]      

    # Remove outliers
    print(">> Removing outliers.")
    for k in X_train_df.columns:
      self.cut_values[k] = [np.quantile(X_train_df.loc[:,k], self.remove_quantile), np.quantile(X_train_df.loc[:,k], 1-self.remove_quantile)]
      select_indices = ((X_train_df.loc[:,k]>=self.cut_values[k][0]) & (X_train_df.loc[:,k]<=self.cut_values[k][1]))
      X_train_df = X_train_df.loc[select_indices,:]
      Y_train_df = Y_train_df.loc[select_indices,:]
      wt_train_df = wt_train_df.loc[select_indices,:]
      select_indices = ((X_test_df.loc[:,k]>=self.cut_values[k][0]) & (X_test_df.loc[:,k]<=self.cut_values[k][1]))
      X_test_df = X_test_df.loc[select_indices,:]
      Y_test_df = Y_test_df.loc[select_indices,:]
      wt_test_df = wt_test_df.loc[select_indices,:]
      select_indices = ((X_val_df.loc[:,k]>=self.cut_values[k][0]) & (X_val_df.loc[:,k]<=self.cut_values[k][1]))
      X_val_df = X_val_df.loc[select_indices,:]
      Y_val_df = Y_val_df.loc[select_indices,:]
      wt_val_df = wt_val_df.loc[select_indices,:]

    # Equalise weights in each unique y values
    if self.equalise_y_wts:
      print(">> Equalising Y weights.")
      wt_train_df = self.EqualiseYWeights(Y_train_df, wt_train_df)
      wt_test_df = self.EqualiseYWeights(Y_test_df, wt_test_df)
      wt_val_df = self.EqualiseYWeights(Y_val_df, wt_val_df)

    # Write parquet files
    print(">> Writing parquet.")
    pq.write_table(pa.Table.from_pandas(X_train_df), self.output_dir+"/X_train.parquet")
    pq.write_table(pa.Table.from_pandas(X_test_df), self.output_dir+"/X_test.parquet")
    pq.write_table(pa.Table.from_pandas(X_val_df), self.output_dir+"/X_val.parquet")
    pq.write_table(pa.Table.from_pandas(Y_train_df), self.output_dir+"/Y_train.parquet")
    pq.write_table(pa.Table.from_pandas(Y_test_df), self.output_dir+"/Y_test.parquet")
    pq.write_table(pa.Table.from_pandas(Y_val_df), self.output_dir+"/Y_val.parquet")
    pq.write_table(pa.Table.from_pandas(wt_train_df), self.output_dir+"/wt_train.parquet")
    pq.write_table(pa.Table.from_pandas(wt_test_df), self.output_dir+"/wt_test.parquet")
    pq.write_table(pa.Table.from_pandas(wt_val_df), self.output_dir+"/wt_val.parquet")

    # Write data parameters
    print(">> Writing parameters yaml.")
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

    if len(Y.columns) > 0:
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
      if column_name in self.parameters["discrete_thresholds"]:
        print(">>>>>>>> Transform discrete to continous.")
        data[column_name] = self.DiscreteToContinuous(data[column_name], column_name, inverse = False)

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
      if column_name in self.parameters["discrete_thresholds"]:
        print(">>>>>>>>> Transform continous to discrete.")
        data[column_name] = self.DiscreteToContinuous(data[column_name], column_name, inverse = True)
          
    return data   

  def UnTransformProb(self, prob, log_prob=False):
    """
    Untransform probabilities.

    Args:
        prob (float): The probability to be untransformed.

    Returns:
        float: The untransformed probability.
    """    
    for column_name in self.parameters["X_columns"]:
      if column_name in self.parameters["standardisation"]:
        if not log_prob:
          prob /= self.parameters["standardisation"][column_name]["std"]
        else:
          prob -= np.log(self.parameters["standardisation"][column_name]["std"])
    return prob 

  def LoadSplitData(self, dataset="train", get=["X","Y","wt"], use_nominal_wt=False):
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
      if not use_nominal_wt:
        wt_dl = DataLoader(f"{self.output_dir}/wt_{dataset}.parquet")
      else:
        wt_dl = DataLoader(f"{self.output_dir}/wt_nominal_{dataset}.parquet")
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
      X_data = X_data.iloc[indices]
      Y_data = Y_data.iloc[indices]
      wt_data = wt_data.iloc[indices]
    
    #X_data = X_data.reindex(range(len(X_data)))
    X_data = X_data.reset_index(drop=True)
    Y_data = Y_data.reset_index(drop=True)
    wt_data = wt_data.reset_index(drop=True)

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

  def PlotY(self, n_bins=20, dataset="train"):
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
