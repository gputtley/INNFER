import os
import copy
import yaml
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from itertools import product
from sklearn.model_selection import train_test_split
from functools import partial

from useful_functions import GetYName, MakeDirectories, GetPOILoop, GetNuisanceLoop
from plotting import plot_histograms
from data_processor import DataProcessor

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
        "selection" : cfg["preprocess"]["selection"] if "selection" in cfg.keys() else None,
        "columns" : cfg["variables"] + cfg["pois"] + cfg["nuisances"] + ["wt"]
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
    # TO DO: Do discrete to continuous stuff

    # Get sum of weights for each unique y combination
    if self.verbose:
      print("- Finding the total yields of entries for specific Y values")
    unique_y_combinations = list(product(*unique_y_values.values()))
    parameters["yield"] = {}
    parameters["effective_events"] = {}
    if len(parameters["Y_columns"]) != 0:
      for uc in unique_y_combinations:
        selection = " & ".join([f"({k}=={uc[ind]})" for ind, k in enumerate(unique_y_values.keys())])
        name = GetYName(uc, purpose="file")
        parameters["yield"][name] = dp.GetFull(method="sum", extra_sel=selection)
        parameters["effective_events"][name] = dp.GetFull(method="n_eff", extra_sel=selection)
    else:
      parameters["yield"]["all"] = dp.GetFull(method="sum")
      parameters["effective_events"]["all"] = dp.GetFull(method="n_eff")

    # Get standardisation parameters
    if self.verbose:
      print("- Finding standardisation parameters")
    parameters["standardisation"] = {}
    means = dp.GetFull(
      method="mean",
      functions_to_apply = [
        partial(self._DoTrainTestValSplit, split="train", train_test_val_split=cfg["preprocess"]["train_test_val_split"])
      ]
    )
    stds = dp.GetFull(
      method="std",
      functions_to_apply = [
        partial(self._DoTrainTestValSplit, split="train", train_test_val_split=cfg["preprocess"]["train_test_val_split"])
      ]
    )    
    for k, v in means.items():
      parameters["standardisation"][k] = {"mean": v, "std": stds[k]}

    # Load and write batches
    # TO DO: Find a way to shuffle the dataset without loading it all into memory
    if self.verbose:
      print("- Making X/Y/wt and train/test/val split datasets and writing them to file")
    dp.parameters = parameters

    # Get yield to scale to
    total_effective_events = dp.GetFull(method="n_eff")
    if len(parameters["Y_columns"]) != 0:
      scale = total_effective_events/len(unique_y_combinations)
    else:
      scale = total_effective_events

    parameters["file_loc"] = self.data_output
    for data_split in ["train","test","val"]:

      for i in ["X","Y","wt"]:
        name = f"{self.data_output}/{i}_{data_split}.parquet"
        MakeDirectories(name)
        if os.path.isfile(name):
          os.system(f"rm {name}")

      dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(self._DoTrainTestValSplit, split=data_split, train_test_val_split=cfg["preprocess"]["train_test_val_split"]),
          partial(self._DoEqualiseYWeights, Y_columns=parameters["Y_columns"], scale_to=scale),
          partial(self._DoDropSpecificYValues, data_split=data_split, train_test_y_vals=cfg["preprocess"]["train_test_y_vals"] if "train_test_y_vals" in cfg["preprocess"].keys() else {}, validation_y_vals=cfg["preprocess"]["validation_y_vals"] if "validation_y_vals" in cfg["preprocess"].keys() else {}),
          "transform",
          partial(self._DoWriteDatasets, X_columns=parameters["X_columns"], Y_columns=parameters["Y_columns"], data_split=data_split)
        ]
      )

    # Delete data processor
    del dp

    # Write parameters file
    if self.verbose:
      print("- Writing parameters yaml")
    with open(self.data_output+"/parameters.yaml", 'w') as yaml_file:
      yaml.dump(parameters, yaml_file, default_flow_style=False)  

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
    for data_split in ["train","test","val"]:
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
        
  def _DoTrainTestValSplit(self, df, split="all", train_test_val_split="0.6:0.1:0.3"):
    train_ratio = float(train_test_val_split.split(":")[0])
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
    if split in ["test","val","all"]:
      test_ratio = float(train_test_val_split.split(":")[1])
      val_ratio = float(train_test_val_split.split(":")[2])   
      val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

    if split == "train":
      return train_df    
    if split == "test":
      return test_df
    elif split == "val":
      return val_df
    else:
      return {"train":train_df, "test":test_df, "val":val_df}
  
  def _DoEqualiseYWeights(self, df, Y_columns=[], scale_to=10000):
    if len(Y_columns) > 0:
      unique_rows = df.loc[:,Y_columns].drop_duplicates()
      for _, ur in unique_rows.iterrows():
        matching_rows = (df.loc[:,Y_columns] == ur).all(axis=1)
        sum_weights = float(np.sum(df.loc[matching_rows,"wt"].to_numpy()))
        df.loc[matching_rows,"wt"] = (scale_to / sum_weights) * df.loc[matching_rows,"wt"]
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

  def _DoDropSpecificYValues(self, df, data_split="train", train_test_y_vals={}, validation_y_vals={}):
    if data_split in ["train","test"]:
      for k, v in train_test_y_vals.items():
        if k in df.columns:
          df = df[df[k].isin(v)]
    if data_split in ["val"]:
      for k, v in validation_y_vals.items():
        if k in df.columns:
          df = df[df[k].isin(v)]
    return df

  def _PlotX(self, vary, freeze, extra_name, parameters, n_bins=40):

    baseline_selection = " & ".join([f"({k}=={v})" for k,v in freeze.items()])
    for data_split in ["train","test", "val"]:
      dp = DataProcessor(
        [[f"{self.data_output}/X_{data_split}.parquet", f"{self.data_output}/Y_{data_split}.parquet", f"{self.data_output}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : None if baseline_selection == "" else baseline_selection,
          "parameters" : parameters
        }
      )
      for transform in [True, False]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]

        unique_values = dp.GetFull(method="unique", functions_to_apply=functions_to_apply)
        for col in parameters["X_columns"]:
          hists = []
          hist_names = []
          _, bins = dp.GetFull(method="histogram", bins=n_bins, functions_to_apply=functions_to_apply, column=col)
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

    for data_split in ["train","test", "val"]:
      dp = DataProcessor(
        [[f"{self.data_output}/Y_{data_split}.parquet", f"{self.data_output}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : None,
          "parameters" : parameters
        }
      )
      for transform in [True, False]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]

        for col in parameters["Y_columns"]:

          hist, bins = dp.GetFull(method="histogram", bins=n_bins, functions_to_apply=functions_to_apply, column=col, discrete_binning=False)

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