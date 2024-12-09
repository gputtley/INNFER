import copy
import gc
import importlib
import os
import pickle
import uproot
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from itertools import product
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split
from scipy.optimize import root_scalar
from scipy.integrate import quad

from data_processor import DataProcessor
from data_loader import DataLoader
from plotting import plot_histograms, plot_spline_and_thresholds
from useful_functions import GetYName, MakeDirectories, GetPOILoop, GetNuisanceLoop, DiscreteTransform
from yields import Yields

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
    self.nuisance = None

    # Other
    self.number_of_shuffles = 10
    self.verbose = True
    self.data_output = "data/"
    self.seed = 2
    self.batch_size = 10**7
    self.discrete_threshold = 50

    # Stores
    self.unique_poi_values = None
    self.yield_df = None
    self.parameters = {}
    self.sum_columns = {}
    self.min_columns = {}
    self.max_columns = {}
    self.discrete_values = {}

  def _DoDiscreteToContinuous(self, data_splits=["train","test"]):

    # Make histograms
    dp = DataProcessor(
      [f"{self.data_output}/X_train.parquet", f"{self.data_output}/wt_train.parquet"],
      "parquet",
      options = {
        "wt_name" : "wt",
      },
      batch_size=self.batch_size,
    )
    unique = dp.GetFull(method="unique")

    # Fit splines
    splines = {}
    thresholds = {}
    integral_differences = {}
    self.parameters["spline_locations"] = {}
    for var, bins in unique.items():
      if bins is None: continue
      bins_for_hist = bins + [2*bins[-1]-bins[-2]] # Maybe fix this??
      hist,_ = dp.GetFull(method="histogram",column=var,bins=bins_for_hist)
      splines[var], thresholds[var], integral_differences[var] = self._DoFitSplineAndFindIntervals(hist, bins)
      spline_loc = f"{self.data_output}/spline_{var}.pkl"
      MakeDirectories(spline_loc)
      with open(spline_loc, 'wb') as file:
        pickle.dump(splines[var], file)
      self.parameters["spline_locations"][var] = spline_loc
      self.min_columns[var] = thresholds[var][bins[0]][0]
      self.max_columns[var] = thresholds[var][bins[-1]][-1]

    self.parameters["discrete_integral_differences"] = integral_differences
    self.parameters["discrete_thresholds"] = thresholds

    # Sample datasets
    for data_split in data_splits:
      name = f"{self.data_output}/X_{data_split}.parquet"
      Y_name = f"{self.data_output}/Y_{data_split}.parquet"
      wt_name = f"{self.data_output}/wt_{data_split}.parquet"
      d2c_name = f"{self.data_output}/X_{data_split}_discrete_to_continuous.parquet"

      # Make data processor
      dp = DataProcessor(
        [name, Y_name, wt_name],
        "parquet",
        batch_size=self.batch_size,
      )

      unique_vals = dp.GetFull(method="unique")
      unique_y_vals = {key:unique_vals[key] for key in self.parameters["Y_columns"]}

      dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(DiscreteTransform, splines=self.parameters["spline_locations"], thresholds=self.parameters["discrete_thresholds"], X_columns=self.parameters["X_columns"], Y_columns=self.parameters["Y_columns"], unique_y_vals=unique_y_vals),
          partial(self._DoWriteFromDataProcesor, file_path=d2c_name)
        ]
      )

      if os.path.isfile(d2c_name):
        os.system(f"mv {d2c_name} {name}")

  def _DoFitSplineAndFindIntervals(self, hist, bins):

    # Bin edges
    bin_min = bins[0] - ((bins[1]-bins[0])/2)
    bin_max = bins[-1] + ((bins[-1]-bins[-2])/2)

    # Fit pdf
    unnormalised_pdf_spline = UnivariateSpline(bins, hist, s=0, k=min(len(bins)-1,3))
    #integral = unnormalised_pdf_spline.integral(bin_min,bin_max)
    integral = quad(unnormalised_pdf_spline, bin_min, bin_max)[0]
    hist_for_normalised_spline = hist/integral
    pdf_spline = UnivariateSpline(bins, hist_for_normalised_spline, s=0, k=min(len(bins)-1,3))    

    # Find threholds for where the pdf integrals equal the cdf values
    cdf = [np.sum(hist[:ind+1]) for ind in range(len(bins))]
    cdf /= cdf[-1]
    intervals = [float(bin_min)]
    for desired_integral in cdf[:-1]:
      #integral_diff = lambda b, desired_integral: pdf_spline.integral(bin_min, b) - desired_integral
      integral_diff = lambda b, desired_integral: quad(pdf_spline, bin_min, b)[0] - desired_integral
      integral_diff = partial(integral_diff, desired_integral=desired_integral)
      result = root_scalar(integral_diff, bracket=[bin_min, bin_max], method='brentq')
      intervals.append(float(result.root))
    intervals.append(float(bin_max))
    
    integral_discrete_on_pdf = np.sum([pdf_spline(b) for b in bins])

    thresholds = {float(bins[ind]) : [intervals[ind],intervals[ind+1]] for ind in range(len(bins))}

    return pdf_spline, thresholds, float(integral_discrete_on_pdf)

  def _DoFlattenYDistributions(self, pois, nuisances, data_splits=["train","test","val"]):

    for data_split in data_splits:

      # Run data processor
      X_name = f"{self.data_output}/X_{data_split}.parquet"
      Y_name = f"{self.data_output}/Y_{data_split}.parquet"
      name = f"{self.data_output}/wt_{data_split}.parquet"
      file_list = [X_name,Y_name,name]
      extra_name = f"{self.data_output}/Extra_{data_split}.parquet"
      if os.path.isfile(extra_name):
        file_list.append(extra_name)

      dp = DataProcessor([file_list],"parquet", batch_size=self.batch_size)

      unique = dp.GetFull(method="unique")

      discrete = True
      for k, v in unique.items():
        if k in pois+nuisances:
          if v is None:
            discrete = False
            break

      if discrete:
        yield_name = f"yields_{data_split}"
      else:
        yield_name = "yield"

      # Build a yield function
      yields_class = Yields(
        pd.read_parquet(self.parameters['yield_loc']), 
        pois, 
        nuisances, 
        self.file_name,
        method="default", 
        column_name=yield_name
      )

      def yield_function(Y,columns):
        Y.loc[:,"wt"] /= yields_class.GetYield(Y.loc[:,columns])
        return Y.loc[:,["wt"]]

      flattened_name = f"{self.data_output}/wt_{data_split}_flattened.parquet"
      if os.path.isfile(flattened_name):
        os.system(f"rm {flattened_name}")   
      dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(yield_function, columns=pois+nuisances),
          partial(self._DoWriteFromDataProcesor, file_path=flattened_name)
        ]
      )
      if os.path.isfile(flattened_name):
        os.system(f"mv {flattened_name} {name}")

  def _DoRemoveOutliers(self, data_splits=["train","test","val"], extra_columns=[]):

    min_selection = " & ".join([f"({k}>={v})" for k,v in self.min_columns.items()])
    max_selection = " & ".join([f"({k}<={v})" for k,v in self.max_columns.items()])
    selection = f"({min_selection}) & ({max_selection})"

    for data_split in data_splits:
      names = [f"{self.data_output}/{i}_{data_split}.parquet" for i in ["X","Y","wt","Extra"] if os.path.isfile(f"{self.data_output}/{i}_{data_split}.parquet")]
      removed_outliers_names = [f"{self.data_output}/{i}_{data_split}_removed_outliers.parquet" for i in ["X","Y","wt","Extra"] if os.path.isfile(f"{self.data_output}/{i}_{data_split}.parquet")]

      dp = DataProcessor(
        [names],
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : selection
        },
        batch_size=self.batch_size,
      )
      dp.parameters = self.parameters
      for removed_outliers_name in removed_outliers_names:
        if os.path.isfile(removed_outliers_name):
          os.system(f"rm {removed_outliers_name}")

      dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(self._DoWriteAllFromDataProcesor, X_columns=self.parameters["X_columns"], Y_columns=self.parameters["Y_columns"], X_file_path=removed_outliers_names[0], Y_file_path=removed_outliers_names[1], wt_file_path=removed_outliers_names[2], extra_file_path=removed_outliers_names[3] if len(removed_outliers_names)>3 else "", extra_columns=extra_columns if len(removed_outliers_names)>3 else [])
        ],
      )

      for ind, removed_outliers_name in enumerate(removed_outliers_names):
        if os.path.isfile(removed_outliers_name):
          os.system(f"mv {removed_outliers_name} {names[ind]}")

  def _DoShuffle(self, data_splits=["train","test"]):

    for data_split in data_splits:
      for i in ["X","Y","wt","Extra"]:
        name = f"{self.data_output}/{i}_{data_split}.parquet"
        if not os.path.isfile(name): continue
        shuffle_name = f"{self.data_output}/{i}_{data_split}_shuffled.parquet"
        if os.path.isfile(shuffle_name):
          os.system(f"rm {shuffle_name}")   
        shuffle_dp = DataProcessor([[name]],"parquet", batch_size=self.batch_size)
        for shuff in range(self.number_of_shuffles):
          print(f" - data_split={data_split}, data_set={i}, shuffle={shuff}")
          shuffle_dp.GetFull(
            method = None,
            functions_to_apply = [
              partial(self._DoShuffleIteration, iteration=shuff, total_iterations=self.number_of_shuffles, seed=42, dataset_name=shuffle_name)
            ]
          )
        if os.path.isfile(shuffle_name):
          os.system(f"mv {shuffle_name} {name}")
        shuffle_dp = DataProcessor([[name]],"parquet", batch_size=self.batch_size)
        shuffle_dp.GetFull(
          method = None,
          functions_to_apply = [
            partial(self._DoShuffleBatch, seed=42, dataset_name=shuffle_name)
          ]
        )
        if os.path.isfile(shuffle_name):
          os.system(f"mv {shuffle_name} {name}")

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

  def _DoShuffleBatch(self, df, seed=42, dataset_name="dataset.parquet"):

    # Select indices
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Write to file
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(dataset_name):
      combined_table = pa.concat_tables([pq.read_table(dataset_name), table])
      pq.write_table(combined_table, dataset_name, compression='snappy')
    else:
      pq.write_table(table, dataset_name, compression='snappy')

    return df

  def _DoStandardisation(self, data_splits=["train","test","val"]):

    dp = DataProcessor(
      [[f"{self.data_output}/{i}_train.parquet" for i in ["X","Y","wt"]]],
      "parquet",
      options = {
        "wt_name" : "wt",
      },
      batch_size=self.batch_size
    )
    means = {k:v/self.sum_columns["wt"] for k,v in self.sum_columns.items() if k!="wt"}
    stds = dp.GetFull(method="std", means=means)
    self.parameters["standardisation"] = {}
    for k, v in means.items():
      self.parameters["standardisation"][k] = {"mean": float(v), "std": float(stds[k])}

    for data_split in data_splits:
      for i in ["X","Y"]:
        name = f"{self.data_output}/{i}_{data_split}.parquet"
        standardised_name = f"{self.data_output}/{i}_{data_split}_standardised.parquet"
        dp = DataProcessor([[name]],"parquet", batch_size=self.batch_size)
        dp.parameters = self.parameters
        if os.path.isfile(standardised_name):
          os.system(f"rm {standardised_name}")   
        dp.GetFull(
          method = None,
          functions_to_apply = [
            "transform",
            partial(self._DoWriteFromDataProcesor, file_path=standardised_name)
          ]
        )
        if os.path.isfile(standardised_name):
          os.system(f"mv {standardised_name} {name}")

  def _DoSumColumns(self, df, columns):

    if len(df) == 0:
      return None
      
    for column in columns:
      if column not in df.columns: continue
      if column == "wt":
        sum_col = np.sum(df.loc[:,"wt"])
      else:
        sum_col = np.sum(df.loc[:,"wt"]*df.loc[:,column])
      if column not in self.sum_columns.keys():
        self.sum_columns[column] = sum_col
      else:
        self.sum_columns[column] += sum_col

  def _DoMinMaxColumns(self, df, columns):

    if len(df) == 0:
      return None
      
    for column in columns:
      if column not in df.columns: continue

      min_col = np.min(df[column])
      max_col = np.max(df[column])

      if column not in self.min_columns.keys():
        self.min_columns[column] = min_col
      elif min_col < self.min_columns[column]:
        self.min_columns[column] = min_col

      if column not in self.max_columns.keys():
        self.max_columns[column] = max_col
      elif max_col > self.max_columns[column]:
        self.max_columns[column] = max_col

  def _DoTrainTestValSplit(self, df, train_test_val_split="0.6:0.1:0.3", train_test_y_vals={}, validation_y_vals={}):

    # Do train/test/val split
    train_ratio = float(train_test_val_split.split(":")[0])
    test_ratio = float(train_test_val_split.split(":")[1])
    val_ratio = float(train_test_val_split.split(":")[2])  
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=self.seed)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), random_state=self.seed)
    test_inf_df = copy.deepcopy(test_df)

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

      # do for test and test inf
      if k in test_df:

        # Find train/test and val keys
        if k not in validation_y_vals.keys():
          v_total = v[:]
          v_inf = []
        else:
          v_total = list(set(v+validation_y_vals[k]))
          v_inf = validation_y_vals[k]

        # Find removed value
        if removed_test_df is None:
          removed_test_df = test_df[~test_df[k].isin(v_total)]
        else:
          removed_test_df = pd.concat([removed_test_df, test_df[~test_df[k].isin(v_total)]])

        # Remove values from test_inf
        test_inf_df = test_inf_df[test_inf_df[k].isin(v_inf)]
        test_df = test_df[test_df[k].isin(v)]

    if removed_train_df is not None:
      if len(removed_train_df) > 0:
        val_df = pd.concat([val_df, removed_train_df])
    if removed_test_df is not None:
      if len(removed_test_df) > 0:
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
      train_ratio = float(train_test_val_split.split(":")[0])
      test_ratio = float(train_test_val_split.split(":")[1])
      sum_ratio = train_ratio + test_ratio
      test_ratio /= sum_ratio
      if len(removed_val_df) > 0:
        train_add_df, test_add_df = train_test_split(removed_val_df, test_size=test_ratio, random_state=42)
        train_df = pd.concat([train_df, train_add_df])
        test_df = pd.concat([test_df, test_add_df])

    return {"train":train_df, "test":test_df, "test_inf":test_inf_df, "val":val_df, "full":copy.deepcopy(df)}

  def _DoWriteAllFromDataProcesor(self, df, X_columns, Y_columns, X_file_path, Y_file_path, wt_file_path, extra_file_path="", extra_columns=[]):
    file_path_translate = {"X":X_file_path, "Y":Y_file_path, "wt":wt_file_path}
    loop_over = {"X":X_columns, "Y":Y_columns, "wt":["wt"]}
    if len(extra_columns) > 0:
      loop_over["Extra"] = [col for col in extra_columns if col in df.columns]
      file_path_translate["Extra"] = extra_file_path
    for data_type, columns in loop_over.items():
      file_path = file_path_translate[data_type]
      table = pa.Table.from_pandas(df.loc[:,columns], preserve_index=False)
      if os.path.isfile(file_path):
        combined_table = pa.concat_tables([pq.read_table(file_path), table])
        pq.write_table(combined_table, file_path, compression='snappy')
      else:
        pq.write_table(table, file_path, compression='snappy')
    return df

  def _DoWriteDatasets(self, df, X_columns, Y_columns, data_splits=["train","test","val"], extra_name="", extra_columns=[]):

    for data_split in data_splits:
      loop_over = {"X":X_columns, "Y":Y_columns, "wt":["wt"]}
      if len(extra_columns) > 0:
        loop_over["Extra"] = [col for col in extra_columns if col in df[data_split].columns]
      for data_type, columns in loop_over.items():
        file_path = f"{self.data_output}{extra_name}/{data_type}_{data_split}.parquet"
        if len(df[data_split]) == 0: continue
        table = pa.Table.from_pandas(df[data_split].loc[:, sorted(columns)], preserve_index=False)
        if os.path.isfile(file_path):
          combined_table = pa.concat_tables([pq.read_table(file_path), table])
          pq.write_table(combined_table, file_path, compression='snappy')
        else:
          pq.write_table(table, file_path, compression='snappy')

  def _DoWriteFromDataProcesor(self, df, file_path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(file_path):
      combined_table = pa.concat_tables([pq.read_table(file_path), table])
      pq.write_table(combined_table, file_path, compression='snappy')
    else:
      pq.write_table(table, file_path, compression='snappy')
    return df

  def _DoWriteParameters(self):
    with open(self.data_output+"/parameters.yaml", 'w') as yaml_file:
      yaml.dump(self.parameters, yaml_file, default_flow_style=False)  
    print(f"Created {self.data_output}/parameters.yaml")

  def _DoWriteYields(self):
    for data_split in ["train", "test", "test_inf", "val"]:
      if f"yields_{data_split}" in self.yield_df.columns or f"sum_wt_squared_{data_split}" in self.yield_df.columns:
        numerator = self.yield_df.loc[:, f"yields_{data_split}"] ** 2
        denominator = self.yield_df.loc[:, f"sum_wt_squared_{data_split}"]
        self.yield_df.loc[:, f"effective_events_{data_split}"] = np.where(denominator != 0, numerator / denominator, 0)
    numerator_total = self.yield_df.loc[:, "yield"] ** 2
    denominator_total = self.yield_df.loc[:, "sum_wt_squared"]
    self.yield_df.loc[:, "effective_events"] = np.where(denominator_total != 0, numerator_total / denominator_total, 0)
    if self.verbose:
      print("- Writing yields yaml")
    self.parameters["yield_loc"] = f"{self.data_output}/yields.parquet"
    MakeDirectories(self.parameters["yield_loc"])
    table = pa.Table.from_pandas(self.yield_df, preserve_index=False)
    pq.write_table(table, self.parameters["yield_loc"], compression='snappy')
    print(f"Created {self.parameters['yield_loc']}")

  def _DoYieldsDataframe(self, df, pois, nuisances, cfg):

    for data_split in ["train", "test", "test_inf", "val", "nuisance_for_yields"]:

      if data_split == "nuisance_for_yields":
        yield_name = "yield"
        sum_wt_squared_name = "sum_wt_squared"
        length_name = "length"
      else:
        yield_name = f"yields_{data_split}"
        sum_wt_squared_name = f"sum_wt_squared_{data_split}"
        length_name = f"length_{data_split}"  

      unique_rows = df[data_split].loc[:,pois+nuisances].drop_duplicates().reset_index(drop=True)
    
      if len(unique_rows) > 100:
        continue

      if len(df[data_split]) == 0:
        continue

      if len(unique_rows) == 0:
        loop = [0]
      else:
        loop = range(len(unique_rows))

      for index in loop:

        if len(unique_rows) > 0:
          row = copy.deepcopy(unique_rows.iloc[[index]])
          filtered_df = df[data_split][(df[data_split].loc[:,row.columns] == row.iloc[0]).all(axis=1)]
        else:
          filtered_df = df[data_split]
        
        sum_wt = np.sum(filtered_df.loc[:,"wt"])
        sum_wt_squared = np.sum(filtered_df.loc[:,"wt"]**2)
        length = len(filtered_df)

        if self.yield_df is None:
          if len(unique_rows) > 0:
            row.loc[:,yield_name] = sum_wt
            row.loc[:,sum_wt_squared_name] = sum_wt_squared
            row.loc[:,length_name] = length
            self.yield_df = copy.deepcopy(row)
          else:
            self.yield_df = pd.DataFrame({yield_name : [sum_wt], sum_wt_squared_name : [sum_wt_squared], length_name : [length]})
        else:
          if len(unique_rows) > 0:
            row_exists = (self.yield_df.loc[:, list(row.columns)] == row.iloc[0]).all(axis=1)
            if not row_exists.any():
              row.loc[:,yield_name] = sum_wt
              row.loc[:,sum_wt_squared_name] = sum_wt_squared
              row.loc[:,length_name] = length
              for col in self.yield_df.columns:
                if col not in row.columns:
                  row.loc[:, col] = 0.0
              self.yield_df.loc[len(self.yield_df)] = row.loc[:, list(self.yield_df.columns)].squeeze()
            else:
              row_index = self.yield_df[row_exists].index[0]
              if not yield_name in self.yield_df.columns:
                self.yield_df.loc[:,yield_name] = 0.0
              if not sum_wt_squared_name in self.yield_df.columns:
                self.yield_df.loc[:,sum_wt_squared_name] = 0.0         
              if not length_name in self.yield_df.columns:
                self.yield_df.loc[:,length_name] = 0.0       
              self.yield_df.loc[row_index, yield_name] += sum_wt
              self.yield_df.loc[row_index, sum_wt_squared_name] += sum_wt_squared 
              self.yield_df.loc[row_index, length_name] += length
          else:
            self.yield_df.loc[0,yield_name] += sum_wt
            self.yield_df.loc[0,sum_wt_squared_name] += sum_wt_squared
            self.yield_df.loc[0,length_name] += length

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

    # Set seed
    np.random.seed(self.seed)

    # Set file names
    self.parameters["file_name"] = self.file_name
    if self.nuisance is not None:
      self.parameters["file_name"] += f"_{self.nuisance}"

    # Set file location
    self.parameters["file_loc"] = self.data_output

    # Open the config
    with open(self.cfg, 'r') as yaml_file:
      cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Set X and Y to parameters
    self.parameters["X_columns"] = sorted(cfg["variables"])

    # Remove dataset if it exists
    for data_split in ["train","test","test_inf","val","full"]:
      for i in ["X","Y","wt","Extra"]:
        name = f"{self.data_output}/{i}_{data_split}.parquet"
        MakeDirectories(name)
        if os.path.isfile(name):
          os.system(f"rm {name}")

    # Loop through input files
    if self.verbose:
      print("- Running event loop")
    for input_file_ind, input_file in enumerate(cfg["files"][self.file_name]["inputs"]):

      if self.verbose:
        print(f"- Loading file {input_file}")

      batch_size = int(os.getenv("EVENTS_PER_BATCH")) if self.batch_size is None else self.batch_size

      if ".root" in input_file:
        # Initiate root file
        file = uproot.open(input_file)
        loader = file[cfg["files"][self.file_name]["tree_name"]]
        total_entries = loader.num_entries
      elif ".parquet" in input_file:
        # Initiate data loader
        loader = DataLoader(input_file,batch_size=batch_size)
        total_entries = loader.num_rows
      else:
        raise ValueError("Input files can only be root or parquet files")

      # Loop through batches
      for start in range(0, total_entries, batch_size):
        stop = min(start + batch_size, total_entries)

        if ".root" in input_file:
          arrays = loader.arrays(
            None,
            cut=cfg["files"][self.file_name]["selection"],
            entry_start=start,
            entry_stop=stop 
          )   
          df = pd.DataFrame(np.array(arrays))
        elif ".parquet" in input_file:
          df = loader.LoadNextBatch()
          if cfg["files"][self.file_name].get("selection", False):
            df = df.loc[df.eval(cfg["files"][self.file_name]["selection"]),:]

        # Skip if dataframe is empty
        if len(df) == 0: continue

        # Add extra columns
        if "add_columns" in cfg["files"][self.file_name].keys():
          for extra_col_name, extra_col_value in cfg["files"][self.file_name]["add_columns"].items():
            df.loc[:,extra_col_name] = extra_col_value[input_file_ind]

        # Calculate weight
        if "weight" in cfg["files"][self.file_name].keys():
          df.loc[:,"wt"] = df.eval(cfg["files"][self.file_name]["weight"])
          df = df.loc[(df.loc[:,"wt"]!=0),:]
        else:
          df.loc[:,"wt"] = 1.0

        # Scale 
        if "scale" in cfg["files"][self.file_name].keys():
          df.loc[:,"wt"] *= cfg["files"][self.file_name]["scale"]

        # Removing nans
        nan_rows = df[df.isna().any(axis=1)]
        if len(nan_rows) > 0:
          df = df.dropna()

        # Set type
        df = df.astype(np.float64)

        # train/test/val split
        df = self._DoTrainTestValSplit(
          df, 
          train_test_val_split=cfg["preprocess"]["train_test_val_split"], 
          train_test_y_vals=cfg["preprocess"]["train_test_y_vals"] if "train_test_y_vals" in cfg["preprocess"].keys() else {}, 
          validation_y_vals=cfg["preprocess"]["validation_y_vals"] if "validation_y_vals" in cfg["preprocess"].keys() else {}
        )        

        # Set up inputs for yield calculations
        df["nuisance_for_yields"] = copy.deepcopy(df["full"])

        # setup shift input dictionary
        shifts = {"pre_calculate" : {}, "weight" : {}, "nuisance_for_yields" : {}}
        nuisance_shifts = []
        if "pre_calculate_shifts" in cfg["files"][self.file_name].keys():
          for k, v in cfg["files"][self.file_name]["pre_calculate_shifts"].items():
            shifts["pre_calculate"][k] = v
            for nui in v.keys():
              if nui in cfg["nuisances"]:
                nuisance_shifts.append(nui)
        if "weight_shifts" in cfg["files"][self.file_name].keys():
          for k, v in cfg["files"][self.file_name]["weight_shifts"].items():
            shifts["weight"][k] = v
            for nui in v.keys():
              if nui in cfg["nuisances"]:
                nuisance_shifts.append(nui)
        shifts["nuisance_for_yields"]["nuisance_for_yields"] = {}
        nuisance_shifts = list(set(nuisance_shifts))
        for nui in nuisance_shifts:
          shifts["nuisance_for_yields"]["nuisance_for_yields"][nui] = {
            "type" : "discrete",
            "values" : [-1.0, 0.0, 1.0],
            "off_axis_with" : None,
          }        

        # setup weight function
        weight_shift_function = "wt"
        if "weight_functions" in cfg["files"][self.file_name].keys():
          for k, v in cfg["files"][self.file_name]["weight_functions"].items():
            weight_shift_function += f"*({v})"

        # setup shift inputs
        for _, shift_type in shifts.items():
          for shift_name, shift_info in shift_type.items():
            if shift_name == "training":
              shift_files = ["train","test"]
            elif shift_name == "nuisance_for_yields":
              shift_files = ["nuisance_for_yields"]
            else:
              shift_files = ["test_inf","val","full"]

            for col, col_info in shift_info.items():

              for data_split in shift_files:

                # setup initial copy
                tmp_copy = copy.deepcopy(df[data_split])
                first = True

                if col_info["type"] == "continuous":
    
                  for ind in range(col_info["n_copies"]):
                    tmp = copy.deepcopy(tmp_copy)
                    tmp.loc[:,col] = np.random.uniform(col_info["range"][0], col_info["range"][1], len(tmp))   
                    if first:
                      df[data_split] = copy.deepcopy(tmp)
                      first = False
                    else:
                      df[data_split] = pd.concat([df[data_split], tmp], axis=0, ignore_index=True)

                elif col_info["type"] == "discrete":
                  
                  # find which columns to shift with
                  if col_info["off_axis_with"] is None:
                    zero_columns = [col for col in cfg["nuisances"] if col in df[data_split].columns]
                  elif col_info["off_axis_with"] == "all":
                    zero_columns = []
                  else:
                    zero_columns = [col for col in col_info["off_axis_with"] if col in df[data_split].columns]

                  # add non shifted events
                  if len(zero_columns) > 0:
                    mask = (tmp_copy[zero_columns] == 0).all(axis=1)
                    df[data_split] = tmp_copy[~mask]
                    first = False
                    tmp_copy = tmp_copy[mask]

                  for val in col_info["values"]:
                    tmp = copy.deepcopy(tmp_copy)
                    tmp.loc[:,col] = val*np.ones(len(tmp))

                    if first:
                      df[data_split] = copy.deepcopy(tmp)
                      first = False
                    else:
                      df[data_split] = pd.concat([df[data_split], tmp], axis=0, ignore_index=True)
                      # check nans and set to 0
                      check_cols = [col for col in cfg["nuisances"] if col in df[data_split].columns]
                      df[data_split].loc[:,check_cols] = df[data_split].loc[:,check_cols].fillna(0)

        for data_split in df.keys():

          if len(df[data_split]) == 0: continue

          # do precalculate if no shift
          for pre_calc_col_name, pre_calc_col_value in cfg["files"][self.file_name]["pre_calculate"].items():
            df[data_split].loc[:,pre_calc_col_name] = df[data_split].eval(pre_calc_col_value)          

          # Apply post selection
          df[data_split] = df[data_split].loc[df[data_split].eval(cfg["files"][self.file_name]["post_calculate_selection"]),:]

          # do weight shift
          df[data_split].loc[:,"wt"] = df[data_split].eval(weight_shift_function)

          # remove y columns
          if "ignore_y_columns" in cfg["preprocess"].keys():
            if self.file_name in cfg["preprocess"]["ignore_y_columns"].keys():
              for y_col in cfg["preprocess"]["ignore_y_columns"][self.file_name]:
                if y_col in df[data_split].columns:
                  df[data_split] = df[data_split].drop(y_col, axis=1)

          # Removing nans
          nan_rows = df[data_split][df[data_split].isna().any(axis=1)]
          if len(nan_rows) > 0:
            df[data_split] = df[data_split].dropna()

        # Only execute on first pass
        if input_file_ind == 0 and start == 0:
          
          # Check nuisances and pois list
          nuisances = [nui for nui in cfg["nuisances"] if nui in df["full"].columns]
          pois = [poi for poi in cfg["pois"] if poi in df["full"].columns]
          extra_columns = cfg["preprocess"]["save_extra_columns"] if "save_extra_columns" in cfg["preprocess"] else []
          extra_columns = [ec for ec in extra_columns if ec in df["full"].columns]

        # Make yield dataframe
        self._DoYieldsDataframe(df, pois, nuisances, cfg)

        # Get the sum of the columns for standardisation parameters
        self._DoSumColumns(df["train"], pois+nuisances+cfg["variables"]+["wt"])

        # Get maximum and minimum values of the training dataset for cutting the other dataset
        self._DoMinMaxColumns(df["train"], cfg["variables"])

        # Remove negative weights
        if "remove_negative_weights" in cfg["preprocess"]:
          if cfg["preprocess"]["remove_negative_weights"]:
            for data_split in ["train","test","test_inf","val"]:
              neg_weight_rows = (df[data_split].loc[:,"wt"] < 0)
              if len(df[data_split][neg_weight_rows]) > 0:
                if self.verbose: 
                  print(f"Total negative weights: {len(df[data_split][neg_weight_rows])}/{len(df[data_split])} = {round(len(df[data_split][neg_weight_rows])/len(df[data_split]),4)}")
                df[data_split] = df[data_split][~neg_weight_rows]

        # Write dataset
        self._DoWriteDatasets(df, cfg["variables"], pois+nuisances, data_splits=["train","test","test_inf","val","full"], extra_name="", extra_columns=cfg["preprocess"]["save_extra_columns"] if "save_extra_columns" in cfg["preprocess"] else [])


    # Set Y and extra columns columns
    self.parameters["Y_columns"] = sorted(pois+nuisances)
    self.parameters["Extra_columns"] = sorted(extra_columns)

    # Make unique Y combinations
    self.parameters["unique_Y_values"] = {}
    for y_key in pois:
      if y_key in cfg["preprocess"]["validation_y_vals"].keys():
        self.parameters["unique_Y_values"][y_key] = cfg["preprocess"]["validation_y_vals"][y_key]
      else:
        raise ValueError(f"Need to specify {y_key} unique values.")
    for y_key in nuisances:
      if y_key in cfg["preprocess"]["validation_y_vals"].keys():
        self.parameters["unique_Y_values"][y_key] = cfg["preprocess"]["validation_y_vals"][y_key]
      else:
        self.parameters["unique_Y_values"][y_key] = [0.0]


    # Make yields
    if self.verbose:
      print("- Writing the yields to file")
    self._DoWriteYields()

    # Normalise each discrete Y value
    if self.verbose:
      print("- Flattening Y distributions")
    self._DoFlattenYDistributions(pois, nuisances, data_splits=["train","test","test_inf","val"])
    
    # Do removing outliers
    if self.verbose:
      print("- Removing outliers")
    self._DoRemoveOutliers(data_splits=["test","test_inf","val"], extra_columns=cfg["preprocess"]["save_extra_columns"] if "save_extra_columns" in cfg["preprocess"] else [])

    # Do discrete to continuous transformation
    #if self.verbose:
    #  print("- Do discrete to continuous")    
    #self._DoDiscreteToContinuous(data_splits=["train","test","test_inf","val"])

    # Do standardisation transformation
    if self.verbose:
      print("- Standardising dataset")
    self._DoStandardisation(data_splits=["train","test","test_inf","val"])

    # Shuffle datasets
    if self.verbose:
      print("- Shuffling the dataset")
    self._DoShuffle(data_splits=["train","test"])

    # Make yields
    if self.verbose:
      print("- Writing the parameters file")
    self._DoWriteParameters()


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [
      self.cfg
    ]
    return inputs
