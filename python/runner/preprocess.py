import copy
import os
import pickle
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

    # Set up data processor
    dp = DataProcessor(
      [[self.parquet_file_name]], 
      "parquet",
      options = {
        "wt_name" : "wt",
        "selection" : cfg["preprocess"]["selection"] if "selection" in cfg["preprocess"].keys() else None,
        "columns" : cfg["variables"] + cfg["pois"] + cfg["nuisances"] + ["wt"],
      }
    )

    # Initiate parameters dictionary
    parameters = {
      "file_name" : self.file_name,
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
    if "validation_y_vals" not in cfg["preprocess"].keys():
      parameters["unique_Y_values"] = copy.deepcopy(unique_y_values)
    else:
      parameters["unique_Y_values"] = cfg["preprocess"]["validation_y_vals"]

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


    # Get sum of weights for each unique y combination
    if self.verbose:
      print("- Finding the total yields of entries for specific Y values and writing to file")
    unique_y_combinations = list(product(*unique_y_values.values()))
    train_test_y_vals = cfg["preprocess"]["train_test_y_vals"] if "train_test_y_vals" in cfg["preprocess"].keys() else {}
    validation_y_vals = cfg["preprocess"]["validation_y_vals"] if "validation_y_vals" in cfg["preprocess"].keys() else {}
    if len(parameters["Y_columns"]) != 0:
      yield_df = pd.DataFrame(unique_y_combinations, columns=parameters["Y_columns"], dtype=np.float64)
      for ind, uc in enumerate(unique_y_combinations):
        if self.verbose:
          print(f" - For Y = {list(uc)}")
        selection = " & ".join([f"({k}=={uc[ind]})" for ind, k in enumerate(unique_y_values.keys())])
        yield_df.loc[ind, "yield"] = dp.GetFull(method="sum", extra_sel=selection)
        yield_df.loc[ind, "effective_events"] = dp.GetFull(method="n_eff", extra_sel=selection)
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

    split_yields_dfs = {}
    unique_y_combinations = list(product(*unique_y_values.values()))
    self.dataset_loop = ["train","test","val"] if cfg["preprocess"]["train_test_val_split"].count(":") == 2 else ["train", "val"]
    for split in self.dataset_loop:
      if self.verbose:
        print(f" - For split {split}")
      if len(parameters["Y_columns"]) != 0:
        split_yields_dfs[split] = pd.DataFrame(unique_y_combinations, columns=parameters["Y_columns"], dtype=np.float64)
        for ind, uc in enumerate(unique_y_combinations):
          if self.verbose:
            print(f"  - For Y = {list(uc)}")
          selection = " & ".join([f"({k}=={uc[ind]})" for ind, k in enumerate(unique_y_values.keys())])
          split_yields_dfs[split].loc[ind, "yield"] = dp.GetFull(
            method="sum", 
            extra_sel=selection,
            functions_to_apply = [
              partial(self._DoTrainTestValSplit, split=split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)
            ]
          )
          yield_df.loc[ind,f"yields_{split}"] = float(split_yields_dfs[split].loc[ind, "yield"])
          yield_df.loc[ind, f"effective_events_{split}"] = dp.GetFull(
            method="n_eff", 
            extra_sel=selection,
            functions_to_apply = [
              partial(self._DoTrainTestValSplit, split=split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)
            ]
          )
      else:
        split_yields_dfs[split] = pd.DataFrame([[dp.GetFull(method="sum", functions_to_apply = [partial(self._DoTrainTestValSplit, split=split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)])]], columns=["yield"], dtype=np.float64)
        yield_df.loc[0,f"yields_{split}"] = float(split_yields_dfs[split].iloc[0].values[0])
        yield_df.loc[0,f"effective_events_{split}"] = dp.GetFull(
          method="n_eff", 
          functions_to_apply = [
            partial(self._DoTrainTestValSplit, split=split, train_test_val_split=cfg["preprocess"]["train_test_val_split"], train_test_y_vals=train_test_y_vals, validation_y_vals=validation_y_vals)
          ]
        )


    # Load and write batches
    # TO DO: Find a way to shuffle the dataset without loading it all into memory
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
          partial(self._DoEqualiseYWeights, yields=split_yields_dfs[data_split], Y_columns=parameters["Y_columns"], scale_to=1.0),
          "transform",
          partial(self._DoWriteDatasets, X_columns=parameters["X_columns"], Y_columns=parameters["Y_columns"], data_split=data_split)
        ]
      )

    # Delete data processor
    del dp

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

    # Run plotting of POIs
    if self.verbose:
      print("- Making plots of POIs")
    for info in GetPOILoop(cfg, parameters):
      self._PlotX(info["poi"], info["freeze"], info["extra_name"], parameters)

    # Run plotting of nuisances
    if self.verbose:
      print("- Making plots of nuisances")
    for info in GetNuisanceLoop(cfg, parameters):
      self._PlotX(info["nuisance"], info["freeze"], info["extra_name"], parameters)

    # Run plotting of Y distributions
    if self.verbose:
      print("- Making plots of Y distributions")
    self._PlotY(parameters)

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
  
  def _DoEqualiseYWeights(self, df, yields, Y_columns=[], scale_to=1.0):
    if len(Y_columns) > 0:
      unique_rows = df.loc[:,Y_columns].drop_duplicates()
      for _, ur in unique_rows.iterrows():
        matching_row = (yields.loc[:,Y_columns] == ur).all(axis=1)
        sum_weights = float(yields.loc[matching_row,"yield"].iloc[0])
        matching_rows = (df.loc[:,Y_columns] == ur).all(axis=1)
        df.loc[matching_rows,"wt"] = (scale_to / sum_weights) * df.loc[matching_rows,"wt"]
    else:
      sum_weights = float(yields.loc[:,"yield"].iloc[0])
      df.loc[:,"wt"] = (scale_to / sum_weights) * df.loc[:,"wt"]      
    return df

  def _DoWriteDatasets(self, df, X_columns=[], Y_columns=[], data_split="train"):

    for data_type, columns in {"X":X_columns, "Y":Y_columns, "wt":["wt"]}.items():
      file_path = f"{self.data_output}/{data_type}_{data_split}.parquet"
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

  def _PlotX(self, vary, freeze, extra_name, parameters, n_bins=40):

    baseline_selection = " & ".join([f"({k}=={v})" for k,v in freeze.items()])
    for data_split in self.dataset_loop:
      dp = DataProcessor(
        [[f"{self.data_output}/X_{data_split}.parquet", f"{self.data_output}/Y_{data_split}.parquet", f"{self.data_output}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : None if baseline_selection == "" else baseline_selection,
          "parameters" : parameters
        }
      )
      for transform in [False, True]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]

        unique_values = dp.GetFull(method="unique", functions_to_apply=functions_to_apply)
        for col in parameters["X_columns"]:
          hists = []
          hist_names = []
          bins = dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, functions_to_apply=functions_to_apply, column=col, ignore_discrete=True)
          if len(parameters["Y_columns"]) != 0:
            for uc in sorted(unique_values[vary]):
              selection = f"({vary}=={uc})"
              hist, bins = dp.GetFull(method="histogram", bins=bins, functions_to_apply=functions_to_apply, extra_sel=selection, column=col)
              hists.append(hist)
              hist_names.append(GetYName([uc], purpose="plot", prefix="y="))
          else:
            hist, bins = dp.GetFull(method="histogram", bins=bins, functions_to_apply=functions_to_apply, column=col)
            hists.append(hist)
            hist_names.append(None)

          extra_name_for_plot = f"{extra_name}_{data_split}"
          if transform:
            extra_name_for_plot += "_transformed"

          plot_name = self.plots_output+f"/X_distributions_varying_{vary}_against_{col}{extra_name_for_plot}"
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

  def _PlotY(self, parameters, n_bins=40):

    for data_split in self.dataset_loop:
      dp = DataProcessor(
        [[f"{self.data_output}/Y_{data_split}.parquet", f"{self.data_output}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : None,
          "parameters" : parameters
        }
      )
      for transform in [False, True]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]

        for col in parameters["Y_columns"]:

          bins = dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, functions_to_apply=functions_to_apply, column=col, discrete_binning=False, ignore_discrete=True)
          hist, bins = dp.GetFull(method="histogram", bins=bins, functions_to_apply=functions_to_apply, column=col, discrete_binning=False)

          extra_name_for_plot = f"{data_split}"
          if transform:
            extra_name_for_plot += "_transformed"
          plot_name = self.plots_output+f"/Y_distributions_for_{col}_{extra_name_for_plot}"
          plot_histograms(
            bins[:-1],
            [hist],
            [None],
            title_right = "",
            name = plot_name,
            x_label = col,
            y_label = "Events",
            anchor_y_at_0 = True,
            drawstyle = "steps-mid",
          )