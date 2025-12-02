import copy
import importlib
import os
import warnings
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor
from useful_functions import (
    FindKeysAndValuesInDictionaries,
    GetDefaults,
    GetDefaultsInModel,
    GetDictionaryEntry,
    GetFilesInModel,
    GetModelLoop,
    GetParametersInModel,
    GetSplitDensityModel,
    GetValidationDefaultIndex,
    GetValidationLoop,
    GetYieldsBaseFile,
    LoadConfig,
    MakeDictionaryEntry,
    MakeDirectories,
    SampleFlatTop
)

from yields import Yields

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

    # Used for parallelisation
    self.partial = None
    self.model_type = None
    self.parameter_name = None
    self.val_ind = None

    # Other
    self.merge_parameters = []
    self.number_of_shuffles = 10
    self.verbose = True
    self.data_input = "data/"
    self.data_output = "data/"
    self.seed = 234
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_PREPROCESS"))
    self.binned_fit_input = None
    self.validation_binned = None
    self.extra_selection = None
    self.category = None
    self.nuisance_shift = None

    # Stores
    self.parameters = {}

  def _ApplyShifts(self, tmp, shifts):
    for k, v in shifts.items():
      if v["type"] == "continuous":
        tmp.loc[:,k] = np.random.uniform(v["range"][0], v["range"][1], size=len(tmp))   
      elif v["type"] == "discrete":
        tmp.loc[:,k] = np.random.choice(v["values"], size=len(tmp))
      elif v["type"] == "fixed":
        tmp.loc[:,k] = v["value"]*np.ones(len(tmp))
      elif v["type"] == "flat_top":
        sigma_out = 0.1*(v["range"][1]-v["range"][0])
        if "other" in v.keys():
          if "sigma_out" in v["other"].keys():
            sigma_out = v["other"]["sigma_out"]
        tmp.loc[:,k] = SampleFlatTop(len(tmp), (v["range"][0], v["range"][1]), sigma_out)
      else:
        raise ValueError(f"Shift type {v['type']} not recognised")
    return tmp


  def _CalculateDatasetWithVariations(self, df, shifts, pre_calculate, post_calculate_selection, weight_shifts, nominal_weight, n_copies=1, post_shifts={}):

    # Distribute the shifts
    tmp_copy = copy.deepcopy(df)
    first = True

    for _ in range(n_copies):
      tmp = self._ApplyShifts(copy.deepcopy(tmp_copy), shifts)
      if first:
        df = copy.deepcopy(tmp)
        first = False
      else:
        df = pd.concat([df, tmp], axis=0, ignore_index=True)

    # do precalculate
    for pre_calc_col_name, pre_calc_col_value in pre_calculate.items():
      if isinstance(pre_calc_col_value, str):
        df.loc[:,pre_calc_col_name] = df.eval(pre_calc_col_value)
      elif isinstance(pre_calc_col_value, dict):
        if pre_calc_col_value["type"] == "function":
          module = importlib.import_module(pre_calc_col_value["file"])
          func = getattr(module, pre_calc_col_value["name"])
          df = func(df, **pre_calc_col_value["args"])
      else:
        raise ValueError(f"Pre calculate type {type(pre_calc_col_value)} not recognised")


    # Apply post selection
    if post_calculate_selection is not None:
      df = df.loc[df.eval(post_calculate_selection),:]

    if self.extra_selection is not None:
      df = df.loc[df.eval(self.extra_selection),:]

    # Post shifts
    df = self._ApplyShifts(copy.deepcopy(df), post_shifts)

    # store old weight before doing weight shift
    df.loc[:,"old_wt"] = df.eval(nominal_weight)

    # do weight shift
    weight_shift_function = nominal_weight
    for k, v in weight_shifts.items():
      if isinstance(v, str):
        weight_shift_function += f"*({v})"        
    df.loc[:,"wt"] = df.eval(weight_shift_function)
    for k, v in weight_shifts.items():
      if isinstance(v, dict):
        if v["type"] == "function":
          module = importlib.import_module(v["file"])
          func = getattr(module, v["name"])
          df = func(df, **v["args"])
        else:
          raise ValueError(f"Weight shift type {v['type']} not recognised")
    df.loc[:,"wt_shift"] = df.loc[:,"wt"]/df.loc[:,"old_wt"]

    # Remove events with 0 weight
    df = df.loc[(df["wt"] != 0), :]

    return df


  def _GetYields(self, file_name, cfg, sigma=1.0, extra_sel=None, base_file_name=None, change_defaults={}):

    # initiate dictionary
    yields = {}

    # Get defaults
    defaults = GetDefaultsInModel(file_name, cfg, include_lnN=True, include_rate=True, category=self.category)
    for k, v in change_defaults.items():
      defaults[k] = v

    # Defaults that are not in the dataset
    if base_file_name is None:
      base_file_name = GetYieldsBaseFile(cfg["models"][file_name]["yields"], self.category)

    parameters_of_file = cfg["files"][base_file_name]["parameters"]
    nominal_shifts = {k: {"type":"fixed","value":v} for k,v in defaults.items() if k not in parameters_of_file}
    if len(parameters_of_file) > 0:
      selection = " & ".join([f"({k}=={defaults[k]})" for k in parameters_of_file])
    else:
      selection = None

    # Build dataprocessor
    dp = DataProcessor(
      [f"{self.data_input}/{base_file_name}.parquet"],
      "parquet",
      options = {
        "wt_name" : "wt",
      },
      batch_size=self.batch_size,
    )

    if cfg["files"][base_file_name]["post_calculate_selection"] is not None and extra_sel is not None:
      post_calculate_selection = f'(({cfg["files"][base_file_name]["post_calculate_selection"]}) & ({extra_sel}))'
    elif cfg["files"][base_file_name]["post_calculate_selection"] is None and extra_sel is not None:
      post_calculate_selection = extra_sel
    elif extra_sel is None and cfg["files"][base_file_name]["post_calculate_selection"] is not None:
      post_calculate_selection = cfg["files"][base_file_name]["post_calculate_selection"]
    else:
      post_calculate_selection = None
    
    # Get nominal
    if self.verbose:
      print(" - Getting nominal yield")
    yields["nominal"] = dp.GetFull(
      method="sum",
      functions_to_apply = [
        partial(
          self._CalculateDatasetWithVariations, 
          shifts=nominal_shifts, 
          pre_calculate=cfg["files"][base_file_name]["pre_calculate"], 
          post_calculate_selection=post_calculate_selection,
          weight_shifts=cfg["files"][base_file_name]["weight_shifts"],
          n_copies=1,
          nominal_weight=cfg["files"][base_file_name]["weight"],
        )
      ],
      extra_sel = selection
    )

    # Initiate log normals
    yields["lnN"] = {}

    # Loop through nuisances
    for nui in cfg["nuisances"]:

      # If yield is 0 then skip
      if yields["nominal"] == 0:
        continue

      # if nuisance in model
      if nui in nominal_shifts.keys():

        up_extra_sel = None
        down_extra_sel = None
        if nui not in parameters_of_file:
          up_shifts = copy.deepcopy(nominal_shifts)
          up_shifts[nui]["value"] = sigma
          down_shifts = copy.deepcopy(nominal_shifts)
          down_shifts[nui]["value"] = -sigma
          if len(parameters_of_file) > 0:
            up_extra_sel = selection
            down_extra_sel = selection
        else:
          up_extra_sel = " & ".join([f"({k}=={defaults[k]})" if k != nui else f"({k}=={sigma})" for k in parameters_of_file])
          down_extra_sel = " & ".join([f"({k}=={defaults[k]})" if k != nui else f"({k}==-{sigma})" for k in parameters_of_file])


        if self.verbose:
          print(f" - Getting yield for nuisance {nui}")

        up_yield = dp.GetFull(
          method="sum",
          functions_to_apply = [
            partial(
              self._CalculateDatasetWithVariations, 
              shifts=up_shifts, 
              pre_calculate=cfg["files"][base_file_name]["pre_calculate"], 
              post_calculate_selection=post_calculate_selection,
              weight_shifts=cfg["files"][base_file_name]["weight_shifts"],
              n_copies=1,
              nominal_weight=cfg["files"][base_file_name]["weight"],
            )
          ],
          extra_sel = up_extra_sel,
        )

        down_yield = dp.GetFull(
          method="sum",
          functions_to_apply = [
            partial(
              self._CalculateDatasetWithVariations, 
              shifts=down_shifts, 
              pre_calculate=cfg["files"][base_file_name]["pre_calculate"], 
              post_calculate_selection=post_calculate_selection,
              weight_shifts=cfg["files"][base_file_name]["weight_shifts"],
              n_copies=1,
              nominal_weight=cfg["files"][base_file_name]["weight"],
            )
          ],
          extra_sel = down_extra_sel,
        )

        yields["lnN"][nui] = [down_yield/yields["nominal"], up_yield/yields["nominal"]]

    # Add log normals that are in the models
    if file_name in cfg["inference"]["lnN"].keys():
      for val in cfg["inference"]["lnN"][file_name]:
        if "categories" not in val.keys() or self.category is None or self.category in val["categories"]:
          if isinstance(val["rate"],list):
            yields["lnN"][val["parameter"]] = val["rate"]
          else:
            yields["lnN"][val["parameter"]] = [1/val["rate"], val["rate"]]

    return yields


  def _CalculateTrainTestValSplit(self, df, file_name="file", splitting="0.8:0.1:0.1", train_test_only=False, validation_only=False, drop_for_training_selection=None, validation_loop_selections={}):

    train_df = None
    test_df = None
    val_df = None
    train_test_df = None

    # get ratios
    train_ratio = float(splitting.split(":")[0])
    test_ratio = float(splitting.split(":")[1])
    val_ratio = float(splitting.split(":")[2])  

    # If don't split return as validation dataset
    if validation_only:

      val_df = copy.deepcopy(df)

    elif train_test_only:

      if drop_for_training_selection is not None:
        train_test_df = df.loc[(df.eval(drop_for_training_selection)),:]
      else:
        train_test_df = copy.deepcopy(df)
      train_df, test_df = train_test_split(train_test_df, test_size=(test_ratio/(train_ratio+test_ratio)))

    else:

      # get potential val df
      if None not in validation_loop_selections:
        total_val_selection = " | ".join(validation_loop_selections)
        potential_val_df = df.loc[(df.eval(total_val_selection)),:]
        train_test_df = df.loc[~(df.eval(total_val_selection)),:]
      else:
        potential_val_df = copy.deepcopy(df)
    
      # make split
      if len(potential_val_df) > 0:
        train_test_from_val_df, val_df = train_test_split(potential_val_df, test_size=val_ratio)          
        del potential_val_df

        # dump out unused train and add to val if possible
        if drop_for_training_selection is not None:
          unused_train_test_df = train_test_from_val_df.loc[(train_test_from_val_df.eval(drop_for_training_selection)),:]
          if None not in validation_loop_selections:
            unused_train_test_df = unused_train_test_df.loc[(unused_train_test_df.eval(total_val_selection)),:]
          if len(unused_train_test_df) > 0:
            val_df = pd.concat([val_df, unused_train_test_df])

        # combine training
        if train_test_df is None:
          train_test_df = copy.deepcopy(train_test_from_val_df)
        else:
          train_test_df = pd.concat([train_test_df, train_test_from_val_df])
        del train_test_from_val_df

      # Remove remaining unused training samples
      if drop_for_training_selection is not None:
        train_test_df = train_test_df.loc[~(train_test_df.eval(drop_for_training_selection)),:]

      # split train and test
      if train_test_df is not None:
        train_df, test_df = train_test_split(train_test_df, test_size=(test_ratio/(train_ratio+test_ratio)))
        
    # Write train and test dataset
    if train_df is not None: self._WriteDataset(train_df, f"{file_name}_train.parquet")
    if test_df is not None: self._WriteDataset(test_df, f"{file_name}_test.parquet")

    # Split validation files (and train and test dataset replicating this)
    if val_df is not None:
      for ind, sel in enumerate(validation_loop_selections):
        if sel is not None:
          val_ind_df = val_df.loc[(val_df.eval(sel)),:]
        else:
          val_ind_df = val_df.copy()
        self._WriteDataset(val_ind_df, f"val_ind_{ind}/{file_name}_val.parquet")
        if train_df is not None:
          if sel is not None:
            train_ind_df = train_df.loc[(train_df.eval(sel)),:]
          else:
            train_ind_df = train_df.copy()
        else:
          train_ind_df = pd.DataFrame(columns=list(val_ind_df.columns))
        self._WriteDataset(train_ind_df, f"val_ind_{ind}/{file_name}_train_inf.parquet")
        if test_df is not None:
          if sel is not None:
            test_ind_df = test_df.loc[(test_df.eval(sel)),:]
          else:
            test_ind_df = test_df.copy()
        else:
          test_ind_df = pd.DataFrame(columns=list(val_ind_df.columns))
        self._WriteDataset(test_ind_df, f"val_ind_{ind}/{file_name}_test_inf.parquet")
        if sel is not None:
          full_ind_df = df.loc[(df.eval(sel)),:]
        else:
          full_ind_df = df.copy()
        self._WriteDataset(full_ind_df, f"val_ind_{ind}/{file_name}_full.parquet")
        del val_ind_df, train_ind_df, test_ind_df, full_ind_df

    return df
    

  def _WriteDataset(self, df, file_name):

    file_path = f"{self.data_output}/{file_name}"
    MakeDirectories(file_path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(file_path):
      combined_table = pa.concat_tables([pq.read_table(file_path), table])
      pq.write_table(combined_table, file_path, compression='snappy')
    else:
      pq.write_table(table, file_path, compression='snappy')

    return df


  def _DoTrainTestValSplit(self, file_name, cfg):

    # get validation loop and their selections
    validation_loop = GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True)

    # get base files
    model_base_files = []
    for density_model in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
      value = cfg["models"][file_name]["density_models"][density_model["loop_index"]]
      model_base_files.append(value["file"])
    for regression_model in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
      value = cfg["models"][file_name]["regression_models"][regression_model["loop_index"]]
      model_base_files.append(value["file"])
    for classifier_model in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
      value = cfg["models"][file_name]["classifier_models"][classifier_model["loop_index"]]
      model_base_files.append(value["file"])
    model_base_files = sorted(list(set(model_base_files)))

    validation_base_files = []
    for value in cfg["validation"]["files"][file_name]:
      if "categories" not in value.keys() or self.category is None or self.category in value["categories"]:
        validation_base_files.append(value["file"])
    validation_base_files = sorted(list(set(validation_base_files)))

    for base_file_name in sorted(list(set(model_base_files+validation_base_files))):

      parameters_in_file = cfg["files"][base_file_name]["parameters"]
      selection_loop = [{k:v for k,v in values.items() if k in parameters_in_file} for values in validation_loop]
      selections = [" & ".join([f"({k}=={v})" for k,v in values.items()]) for values in selection_loop]
      selections = [v if not v=="" else None for v in selections]

      # get unused training selection
      if "drop_from_training" in cfg["preprocess"].keys():
        unused_training_selection = " | ".join([" | ".join([f"{k}=={val}"for val in v]) for k, v in cfg["preprocess"]["drop_from_training"][file_name].items()])
        unused_training_selection = None if unused_training_selection == "" else unused_training_selection
      else:
        unused_training_selection = None

      # Delete files
      for i in ["train","test"]:
        outfile = f"{self.data_output}/{base_file_name}_{i}.parquet"
        if os.path.isfile(outfile):
          os.system(f"rm {outfile}")
      for i in ["train_inf","test_inf", "val","full"]:
        for ind in range(len(selections)):
          outfile = f"{self.data_output}/val_ind_{ind}/{base_file_name}_{i}.parquet"
          if os.path.isfile(outfile):
            os.system(f"rm {outfile}")

      # Run train test validation on common file split
      dp = DataProcessor(
        [f"{self.data_input}/{base_file_name}.parquet"],
        "parquet",
        options = {
          "wt_name" : "wt",
        },
        batch_size=self.batch_size,
      )
      
      dp.GetFull(
        method=None,
        functions_to_apply = [
          partial(
            self._CalculateTrainTestValSplit, 
            file_name=base_file_name, 
            splitting=cfg["preprocess"]["train_test_val_split"], 
            validation_only=(base_file_name not in model_base_files),
            train_test_only=(base_file_name not in validation_base_files), 
            drop_for_training_selection=unused_training_selection, 
            validation_loop_selections=selections
          )
        ]
      )


  def _WriteSplitDataset(self, df, extra_dir, extra_name, split_dict):

    file_path = f"{self.data_output}/{extra_dir}/"
    MakeDirectories(file_path)
    for k, v in split_dict.items():
      file_name = f"{file_path}/{k}_{extra_name}.parquet"

      skimmed_df = df.loc[:,v]
      if k == "wt":
        skimmed_df = skimmed_df.rename(columns={v[0]:"wt"})

      table = pa.Table.from_pandas(skimmed_df, preserve_index=False)
      if os.path.isfile(file_name):
        combined_table = pa.concat_tables([pq.read_table(file_name), table])
        pq.write_table(combined_table, file_name, compression='snappy')
      else:
        pq.write_table(table, file_name, compression='snappy')
    return df


  def _DoWriteModelVariation(self, value, directory, file_name, cfg, extra_dir, extra_name, split_dict):

    dp = DataProcessor(
      [f"{directory}/{file_name}.parquet"],
      "parquet",
      options = {
        "wt_name" : "wt",
      },
      batch_size=int(np.ceil(self.batch_size/value["n_copies"])),
    )

    if np.sum(dp.num_batches) == 0:

      columns = []
      for val in split_dict.values():
        columns += val
      columns = sorted(list(set(columns)))

      self._WriteSplitDataset(
        pd.DataFrame(columns=columns),
        extra_dir=extra_dir, 
        extra_name=extra_name, 
        split_dict=split_dict          
      )

    else:

      base_file_name = value['file']
      dp.GetFull(
        method=None,
        functions_to_apply = [
          partial(
            self._CalculateDatasetWithVariations, 
            shifts=value["shifts"], 
            pre_calculate=cfg["files"][base_file_name]["pre_calculate"], 
            post_calculate_selection=cfg["files"][base_file_name]["post_calculate_selection"], 
            weight_shifts=cfg["files"][base_file_name]["weight_shifts"],
            n_copies=value["n_copies"],
            nominal_weight=cfg["files"][base_file_name]["weight"],
            post_shifts=value["post_shifts"] if "post_shifts" in value.keys() else {}
          ),
          partial(
            self._WriteSplitDataset, 
            extra_dir=extra_dir, 
            extra_name=extra_name, 
            split_dict=split_dict
          )
        ]
      )


  def _DoReweightToShift(self, col_files, wt_file, shifts, selection=None, n_bins=40, samples=10**7):

    rdp = DataProcessor(
      [[f"{self.data_output}/{i}" for i in col_files] + [f"{self.data_output}/{wt_file}"]],
      "parquet",
      batch_size=self.batch_size,
      options = {
        "wt_name" : "wt",
      }      
    )

    for k,v in shifts.items():

      if v["type"] not in ["continuous","flat_top"]:
        continue
    
      bins = rdp.GetFull(method="bins_with_equal_stats", column=k, bins=n_bins, ignore_quantile=0.0)
      hist, _ = rdp.GetFull(method="histogram", column=k, bins=bins, ignore_quantile=0.0, extra_sel=selection)

      # Check what distribution we want to flatten to
      if v["type"] == "continuous":
        hist_to = (np.sum(hist)/len(hist))*np.ones(len(hist))
      elif v["type"] == "flat_top":
        samples = SampleFlatTop(samples, (v["range"][0], v["range"][1]), v["other"]["sigma_out"])
        hist_to, _ = np.histogram(samples, bins=bins)
        hist_to = hist_to.astype(float)
        hist_to *= np.sum(hist)/np.sum(hist_to)

      # Cap spline to boundaries
      spline = CubicSpline((bins[1:]+bins[:-1])/2, hist_to/hist, extrapolate=True, bc_type='clamped')

      wt_reweight_name = wt_file.replace(".parquet",f"_{k}_reweight.parquet")

      if os.path.isfile(f"{self.data_output}/{wt_reweight_name}"):
        os.system(f"rm {self.data_output}/{wt_reweight_name}")

      def ApplySpline(df, spline, k, selection=None):
        if selection is not None:
          mask = df.eval(selection)
        else:
          mask = np.ones(len(df), dtype=bool)
        df.loc[mask,"wt"] *= spline(df.loc[mask,k])
        return df.loc[:,["wt"]]


      rdp.GetFull(
        method=None,
        functions_to_apply = [
          partial(
            ApplySpline, 
            spline=spline, 
            k=k,
            selection=selection
          ),
          partial(
            self._WriteDataset, 
            file_name=wt_reweight_name
          )
        ],
      )

      os.system(f"mv {self.data_output}/{wt_reweight_name} {self.data_output}/{wt_file}")


  def _DoModelVariations(self, file_name, cfg, do_clear=True):

    # Check if we need to split density models
    split_density_model = GetSplitDensityModel(cfg, file_name, category=self.category)

    density_split_model = {}

    # Get extra columns
    extra_cols = None
    if "save_extra_columns" in cfg["preprocess"]:
      if file_name in cfg["preprocess"]["save_extra_columns"]:
        extra_cols = cfg["preprocess"]["save_extra_columns"][file_name]
        density_split_model["Extra"] = extra_cols

    # Delete files
    for data_split in ["train","test"]:
      for k in ["X","Y","wt","Extra"]:

        if self.model_type is None or self.model_type == "density_models":
          for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
            value = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
            if not split_density_model:
              outfile = f"{self.data_output}/density/{k}_{data_split}.parquet"
            else:
              outfile = f"{self.data_output}/density/split_{value['split']}/{k}_{data_split}.parquet"
            if os.path.isfile(outfile):
              os.system(f"rm {outfile}")

        if self.model_type is None or self.model_type == "regression_models":
          for k in ["X","y","wt","Extra"]:
            for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
              value = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
              if self.parameter_name is None or self.parameter_name == value["parameter"]:
                outfile = f"{self.data_output}/regression/{value['parameter']}/{k}_{data_split}.parquet"
                if os.path.isfile(outfile):
                  os.system(f"rm {outfile}")

        if self.model_type is None or self.model_type == "classifier_models":
          for k in ["X","y","wt","Extra"]:
            for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
              value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
              if self.parameter_name is None or self.parameter_name == value["parameter"]:
                outfile = f"{self.data_output}/classifier/{value['parameter']}/{k}_{data_split}.parquet"
                if os.path.isfile(outfile):
                  os.system(f"rm {outfile}")

    # Get defaults
    defaults = GetDefaults(cfg)

    # Loop through the train and test
    for data_split in ["train","test"]:

      # Do density models
      if self.model_type is None or self.model_type == "density_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
          if self.verbose:
            print(f" - Processing density model variation for {file_name}, split: {data_split}, model index: {loop_value['loop_index']}")
          value = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
          value_copy = copy.deepcopy(value)
          for k, v in defaults.items():
            if k not in value["parameters"]:
              value_copy["shifts"][k] = {"type":"fixed","value":v}
          density_split_model_val = {"X":cfg["variables"],"Y":value["parameters"],"wt":["wt"]}
          for k, v in density_split_model.items(): density_split_model_val[k] = v
          if not split_density_model:
            self._DoWriteModelVariation(value_copy, self.data_output, f"{value['file']}_{data_split}", cfg, "density", data_split, split_dict=density_split_model_val)
          else:
            self._DoWriteModelVariation(value_copy, self.data_output, f"{value['file']}_{data_split}", cfg, f"density/split_{value['split']}", data_split, split_dict=density_split_model_val)

      # Do regression models
      if self.model_type is None or self.model_type == "regression_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
          value = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
          if self.parameter_name is None or self.parameter_name == value["parameter"]:
            if self.verbose:
              print(f" - Processing regression model variation for {file_name}, split: {data_split}, parameter: {value['parameter']}")
            value_copy = copy.deepcopy(value)
            for k, v in defaults.items():
              if k != value["parameter"]:
                value_copy["shifts"][k] = {"type":"fixed","value":v}
            regression_split_model = {"X":cfg["variables"]+[value["parameter"]], "y":["wt_shift"], "wt":["old_wt"]}
            if extra_cols is not None:
              regression_split_model["Extra"] = extra_cols
            self._DoWriteModelVariation(value_copy, self.data_output, f"{value['file']}_{data_split}", cfg, f"regression/{value['parameter']}", data_split, split_dict=regression_split_model)

      # Do classifier models
      if self.model_type is None or self.model_type == "classifier_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
          value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
          if self.parameter_name is None or self.parameter_name == value["parameter"]:
            if self.verbose:
              print(f" - Processing classifier model variation for {file_name}, split: {data_split}, parameter: {value['parameter']}")
            value_copy = copy.deepcopy(value)
            for k, v in defaults.items():
              if k != value["parameter"]:
                value_copy["shifts"][k] = {"type":"fixed","value":v}
            value_copy["shifts"]["classifier_truth"] = {"type":"fixed","value":1.0}
            classifier_split_model = {"X":cfg["variables"]+[value["parameter"]], "y":["classifier_truth"], "wt":["wt"]}
            if extra_cols is not None:
              classifier_split_model["Extra"] = extra_cols
            self._DoWriteModelVariation(value_copy, self.data_output, f"{value['file']}_{data_split}", cfg, f"classifier/{value['parameter']}", data_split, split_dict=classifier_split_model)
            value_default = copy.deepcopy(value)
            for k, v in defaults.items():
              value_default["shifts"][k] = {"type":"fixed","value":v}
            value_default["shifts"]["classifier_truth"] = {"type":"fixed","value":0.0}
            value_default["post_shifts"] = copy.deepcopy(value_copy["shifts"])
            value_default["post_shifts"]["classifier_truth"] = {"type":"fixed","value":0.0}
            self._DoWriteModelVariation(value_default, self.data_output, f"{value['file']}_{data_split}", cfg, f"classifier/{value['parameter']}", data_split, split_dict=classifier_split_model)

    if do_clear:
      # Clear up old files
      for k in cfg["files"].keys():
        for data_split in ["train","test"]:
          outfile = f"{self.data_output}/{k}_{data_split}.parquet"
          if os.path.isfile(outfile):
            os.system(f"rm {outfile}")
  

  def _DoShiftReweighting(self, file_name, cfg):

    # Check if we need to split density models
    split_density_model = GetSplitDensityModel(cfg, file_name, category=self.category)

    # Get defaults
    defaults = GetDefaults(cfg)

    # Loop through the train and test
    for data_split in ["train","test"]:

      # Do density models
      if self.model_type is None or self.model_type == "density_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
          value = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
          value_copy = copy.deepcopy(value)
          for k, v in defaults.items():
            if k not in value["parameters"]:
              value_copy["shifts"][k] = {"type":"fixed","value":v}
          if not split_density_model:
            self._DoReweightToShift([f"density/{i}_{data_split}.parquet" for i in ["X","Y"]], f"density/wt_{data_split}.parquet", value_copy["shifts"])
          else:
            self._DoReweightToShift([f"density/split_{value['split']}/{i}_{data_split}.parquet" for i in ["X","Y"]], f"density/split_{value['split']}/wt_{data_split}.parquet", value_copy["shifts"])

      # Do regression models
      if self.model_type is None or self.model_type == "regression_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
          value = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
          if self.parameter_name is None or self.parameter_name == value["parameter"]:
            value_copy = copy.deepcopy(value)
            for k, v in defaults.items():
              if k != value["parameter"]:
                value_copy["shifts"][k] = {"type":"fixed","value":v}
            self._DoReweightToShift([f"regression/{value['parameter']}/{i}_{data_split}.parquet" for i in ["X","y"]], f"regression/{value['parameter']}/wt_{data_split}.parquet", value_copy["shifts"])

      # Do classifier models
      if self.model_type is None or self.model_type == "classifier_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
          value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
          if self.parameter_name is None or self.parameter_name == value["parameter"]:
            value_copy = copy.deepcopy(value)
            for k, v in defaults.items():
              if k != value["parameter"]:
                value_copy["shifts"][k] = {"type":"fixed","value":v}
            value_copy["shifts"]["classifier_truth"] = {"type":"fixed","value":1.0}
            self._DoReweightToShift([f"classifier/{value['parameter']}/{i}_{data_split}.parquet" for i in ["X","y"]], f"classifier/{value['parameter']}/wt_{data_split}.parquet", value_copy["shifts"], selection="(classifier_truth==0.0)")
            self._DoReweightToShift([f"classifier/{value['parameter']}/{i}_{data_split}.parquet" for i in ["X","y"]], f"classifier/{value['parameter']}/wt_{data_split}.parquet", value_copy["shifts"], selection="(classifier_truth==1.0)")

  def _DoValidationVariations(self, file_name, cfg, do_clear=True):

    # Get extra columns
    extra_cols = None
    if "save_extra_columns" in cfg["preprocess"]:
      if file_name in cfg["preprocess"]["save_extra_columns"]:
        extra_cols = cfg["preprocess"]["save_extra_columns"][file_name]

    # Get defaults
    defaults = GetDefaults(cfg)

    for data_split in ["val","train_inf","test_inf","full"]:
      for k in ["X","Y","wt","Extra"]:
        for ind in range(len(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True))):
          if self.val_ind is None or self.val_ind == ind:
            outfile = f"{self.data_output}/val_ind_{ind}/{k}_{data_split}.parquet"
            if os.path.isfile(outfile):
              os.system(f"rm {outfile}")      

    # Loop through validation files
    for data_split in ["val","train_inf","test_inf","full"]:
      for ind, val in enumerate(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True)):
        if self.val_ind is None or self.val_ind == ind:

          base_file_name = None
          for value in cfg["validation"]["files"][file_name]:
            if "categories" not in value.keys() or self.category is None or self.category in value["categories"]:
              base_file_name = value["file"]
              break
          parameters_in_file = cfg["files"][base_file_name]["parameters"]
          shift_parameters = [k for k in val.keys() if k not in parameters_in_file]
          shifts = {}
          for k, v in defaults.items():
            if k not in parameters_in_file:
              shifts[k] = {"type":"fixed","value":v}
          for k in shift_parameters:
            shifts[k] = {"type":"fixed","value":val[k]}
          val_shift = {
            "parameters" : GetParametersInModel(file_name, cfg, category=self.category),
            "file" : base_file_name,
            "n_copies" : 1,
            "shifts" : shifts
          }
          val_split_model = {"X":cfg["variables"], "Y":val_shift["parameters"], "wt":["wt"]}
          if extra_cols is not None:
            val_split_model["Extra"] = extra_cols 
          self._DoWriteModelVariation(val_shift, self.data_output, f"val_ind_{ind}/{base_file_name}_{data_split}", cfg, f"val_ind_{ind}", data_split, split_dict=val_split_model)


    if do_clear:
      # Clear up old files
      default_index = GetValidationDefaultIndex(cfg, file_name)
      for k in cfg["files"].keys():
        for data_split in ["val","train_inf","test_inf","full"]:
          for ind in range(len(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True))):
            if ind == default_index: continue
            if self.val_ind is None or self.val_ind == ind:
              outfile = f"{self.data_output}/val_ind_{ind}/{k}_{data_split}.parquet"
              if os.path.isfile(outfile):
                os.system(f"rm {outfile}")    


  def _DoValidationNormalisation(self, file_name, cfg, yields):

    for data_split in ["val","train_inf","test_inf","full"]:

      for ind, val_loop in enumerate(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True)):

        if self.val_ind is None or self.val_ind == ind:

          outfile = f"val_ind_{ind}/wt_{data_split}_norm.parquet"
          nomfile = f"val_ind_{ind}/wt_{data_split}.parquet"

          # Delete file
          if os.path.isfile(outfile):
            os.system(f"rm {outfile}")

          # Build dataprocessor
          dp = DataProcessor(
            [[f"{self.data_output}/{nomfile}"]],
            "parquet",
            options = {
              "wt_name" : "wt",
            },
            batch_size=self.batch_size,
          )

          # Get normalisation
          if np.sum(dp.num_batches) == 0:
            continue

          norm = dp.GetFull(method="sum")

          def normalisation(df, norm):
            if len(df) == 0: 
              return df
            df.loc[:,"wt"] /= norm
            return df

          yield_class = Yields(
            yields["nominal"],
            lnN = yields["lnN"],
            physics_model = None,
            rate_param = f"mu_{file_name}" if file_name in cfg["inference"]["rate_parameters"] else None,
          )

          params_in_model = GetDefaultsInModel(file_name, cfg, include_rate=True, include_lnN=True, category=self.category)
          for k, v in val_loop.items():
            params_in_model[k] = v

          scaler = yield_class.GetYield(pd.DataFrame({k:[v] for k, v in params_in_model.items()}))

          # Normalise dataset
          dp.GetFull(
            method=None,
            functions_to_apply = [
              partial(normalisation, norm=norm/scaler),
              partial(self._WriteDataset, file_name=outfile)
            ]
          )

          # Move over
          os.system(f"mv {self.data_output}/{outfile} {self.data_output}/{nomfile}")
    


  def _DoNuisanceNormalisation(self, file_name, cfg, yields):

    for data_split in ["val","train_inf","test_inf","full"]:

      # Get defaults
      params_in_model = GetDefaultsInModel(file_name, cfg, include_rate=True, include_lnN=True, category=self.category)

      for nui in cfg["nuisances"]:

        if nui not in params_in_model.keys():
          continue

        for shift in ["up","down"]:
          
          val_key = f"{nui}_{shift}"

          if self.nuisance_shift is None or self.nuisance_shift == val_key:

            outfile = f"{val_key}/wt_{data_split}_norm.parquet"
            nomfile = f"{val_key}/wt_{data_split}.parquet"

            # Delete file
            if os.path.isfile(outfile):
              os.system(f"rm {outfile}")

            # Build dataprocessor
            dp = DataProcessor(
              [[f"{self.data_output}/{nomfile}"]],
              "parquet",
              options = {
                "wt_name" : "wt",
              },
              batch_size=self.batch_size,
            )

            # Get normalisation
            if np.sum(dp.num_batches) == 0:
              continue

            norm = dp.GetFull(method="sum")

            def normalisation(df, norm):
              if len(df) == 0: 
                return df
              df.loc[:,"wt"] /= norm
              return df

            yield_class = Yields(
              yields["nominal"],
              lnN = yields["lnN"],
              physics_model = None,
              rate_param = f"mu_{file_name}" if file_name in cfg["inference"]["rate_parameters"] else None,
            )

            Y_vals = copy.deepcopy(params_in_model)
            Y_vals[nui] = 1.0 if shift == "up" else -1.0

            scaler = yield_class.GetYield(pd.DataFrame({k:[v] for k, v in Y_vals.items()}))
            # Normalise dataset
            dp.GetFull(
              method=None,
              functions_to_apply = [
                partial(normalisation, norm=norm/scaler),
                partial(self._WriteDataset, file_name=outfile)
              ]
            )

            # Move over
            os.system(f"mv {self.data_output}/{outfile} {self.data_output}/{nomfile}")



  def _GetBinnedFitInputs(self, file_name, cfg, yields):

    # Check if binned fit is needed
    if "binned_fit" not in cfg["inference"].keys():
      return None

    if cfg["inference"]["binned_fit"] == {}:
      return None

    # Check that there are not too many shape POIs
    if len(cfg["pois"]) > 1:
      raise ValueError("Binned fits are not setup to work for more the one shape POI.")

    # Get the parameters in the model
    parameters_in_model = GetParametersInModel(file_name, cfg, category=self.category)

    # Need to scale to the nominal yield function
    if not (len(cfg["pois"]) == 0 or cfg["pois"][0] not in parameters_in_model):
      total_scales = {}
      for poi_val in cfg["inference"]["binned_fit"]["shape_poi_values"]:
        poi_yields = self._GetYields(
          file_name,
          cfg,
          extra_sel=None,
          base_file_name=cfg["inference"]["binned_fit"]["files"][self.file_name],
          change_defaults={cfg["pois"][0]: poi_val},
        )
        total_scales[poi_val] = yields["nominal"]/poi_yields["nominal"]

    # Loop through bins
    inputs_for_binned_fit = []
    for bin_ind in range(len(cfg["inference"]["binned_fit"]["input"][self.category]["binning"])-1):

      cat = cfg["inference"]["binned_fit"]["input"][self.category]

      # Get selection
      bin_sel = f"({cat['variable']}>={cat['binning'][bin_ind]}) & ({cat['variable']}<{cat['binning'][bin_ind+1]})"

      # Setup bins
      inputs_for_binned_fit.append({
        "bin_ind" : bin_ind,
        "selection" : bin_sel,
        "yields" : {}
      })

      # Make histograms
      if len(cfg["pois"]) == 0 or cfg["pois"][0] not in parameters_in_model:
        
        inputs_for_binned_fit[-1]["yields"]["all"] = self._GetYields(
          file_name,
          cfg,
          extra_sel=bin_sel,
          base_file_name=cfg["inference"]["binned_fit"]["files"][self.file_name],
        )

      else:

        for poi_val in cfg["inference"]["binned_fit"]["shape_poi_values"]:
          inputs_for_binned_fit[-1]["yields"][poi_val] = self._GetYields(
            file_name,
            cfg,
            extra_sel=bin_sel,
            base_file_name=cfg["inference"]["binned_fit"]["files"][self.file_name],
            change_defaults={cfg["pois"][0]: poi_val},
          )
          inputs_for_binned_fit[-1]["yields"][poi_val]["nominal"] *= total_scales[poi_val]

    self.binned_fit_input = inputs_for_binned_fit



  def _GetValidationBinned(self, file_name, cfg):

    # Check if binned fit is needed
    if "binned_fit" not in cfg["inference"].keys():
      return None
    if cfg["inference"]["binned_fit"] == {}:
      return None

    self.validation_binned = {}
    for data_split in ["val","train_inf","test_inf","full"]:
      self.validation_binned[data_split] = {}
      for ind, _ in enumerate(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True)):

        if self.val_ind is None or self.val_ind == ind:

          loop = ["X","Y","wt"]
          if "save_extra_columns" in cfg["preprocess"]:
            if self.file_name in cfg["preprocess"]["save_extra_columns"]:
              loop += ["Extra"]

          # Build dataprocessor
          dp = DataProcessor(
            [[f"{self.data_output}/val_ind_{ind}/{i}_{data_split}.parquet" for i in loop]],
            "parquet",
            options = {
              "wt_name" : "wt",
            },
            batch_size=self.batch_size,
          )

          # Get normalisation
          if np.sum(dp.num_batches) == 0:
            continue

          self.validation_binned[data_split][ind] = []
          for bin_ind in range(len(cfg["inference"]["binned_fit"]["input"][self.category]["binning"])-1):
            cat = cfg["inference"]["binned_fit"]["input"][self.category]

            # Get selection
            bin_sel = f"({cat['variable']}>={cat['binning'][bin_ind]}) & ({cat['variable']}<{cat['binning'][bin_ind+1]})"
            self.validation_binned[data_split][ind].append(
              dp.GetFull(
                method="sum",
                extra_sel=bin_sel,
              )
            )


  def _GetValidationEffEvents(self, file_name, cfg):

    eff_events = {}
    for data_split in ["val","train_inf","test_inf","full"]:
      eff_events[data_split] = {}
      for ind in range(len(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True))):

        if self.val_ind is None or self.val_ind == ind:

          nomfile = f"val_ind_{ind}/wt_{data_split}.parquet"

          # Build dataprocessor
          dp = DataProcessor(
            [[f"{self.data_output}/{nomfile}"]],
            "parquet",
            options = {
              "wt_name" : "wt",
            },
            batch_size=self.batch_size,
          )

          if np.sum(dp.num_batches) == 0:
            eff_events[data_split][ind] = 0.0
            continue

          # Get normalisation
          eff_events[data_split][ind] = dp.GetFull(method="n_eff")

    return eff_events


  def _GetStandardisationParameters(self, file_name, cfg):

    standardisation_parameters = {}

    # Check if we need to split density models
    split_density_model = GetSplitDensityModel(cfg, file_name, category=self.category)

    # density model
    if self.model_type is None or self.model_type == "density_models":
      standardisation_parameters["density"] = {}
      if not split_density_model:

        sp = GetDictionaryEntry(cfg["preprocess"], ["standardisation",self.file_name,self.category,"density"])
        if sp is None:
          density_files = [[f"{self.data_output}/density/X_train.parquet", f"{self.data_output}/density/Y_train.parquet", f"{self.data_output}/density/wt_train.parquet"]]
          dp = DataProcessor(
            density_files,
            "parquet",
            options = {
              "wt_name" : "wt",
            },
            batch_size=self.batch_size,
          )
          density_means = dp.GetFull(method="mean")
          density_stds = dp.GetFull(method="std")
          for col in density_means.keys():
            standardisation_parameters["density"][col] = {
              "mean" : density_means[col],
              "std" : density_stds[col],
            }
            
        else:

          for col in sp.keys():
            standardisation_parameters["density"][col] = {
              "mean" : sp[col]["mean"],
              "std" : sp[col]["std"],
            }

      else:

        # Get X standardisation parameters
        sp = GetDictionaryEntry(cfg["preprocess"], ["standardisation",self.file_name,self.category,"density"])
        if sp is None:
          split_density_files = []
          for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
            value = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
            if loop_value["split"] not in split_density_files:
              split_density_files.append(value["split"])
          density_files = [[f"{self.data_output}/density/split_{ind}/X_train.parquet", f"{self.data_output}/density/split_{ind}/wt_train.parquet"] for ind in split_density_files]
          dp = DataProcessor(
            density_files,
            "parquet",
            options = {
              "wt_name" : "wt",
            },
            batch_size=self.batch_size,
          )
          density_means = dp.GetFull(method="mean")
          density_stds = dp.GetFull(method="std")
          for col in cfg["variables"]:
            standardisation_parameters["density"][col] = {
              "mean" : density_means[col],
              "std" : density_stds[col],
            }

        # Get Y standardisation parameters
        for col in GetParametersInModel(file_name, cfg, only_density=True, category=self.category):
          density_files = []
          for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
            value = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
            ind = loop_value["split"]
            if col in value["parameters"]:
              density_files.append([f"{self.data_output}/density/split_{ind}/Y_train.parquet", f"{self.data_output}/density/split_{ind}/wt_train.parquet"])
          dp = DataProcessor(
            density_files,
            "parquet",
            options = {
              "wt_name" : "wt",
            },
            batch_size=self.batch_size,
          )
          density_means = dp.GetFull(method="mean")
          density_stds = dp.GetFull(method="std")
          standardisation_parameters["density"][col] = {
            "mean" : density_means[col],
            "std" : density_stds[col],
          }

        else:

          for col in sp.keys():
            standardisation_parameters["density"][col] = {
              "mean" : sp[col]["mean"],
              "std" : sp[col]["std"],
            }


    # regression models
    if self.model_type is None or self.model_type == "regression_models":
      standardisation_parameters["regression"] = {}
      for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
        value = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
        name = value["parameter"]
        if self.parameter_name is None or self.parameter_name == name:
          sp = GetDictionaryEntry(cfg["preprocess"], ["standardisation",self.file_name,self.category,"regression",name])

          if sp is None:
            dp = DataProcessor(
              [[f"{self.data_output}/regression/{name}/X_train.parquet", f"{self.data_output}/regression/{name}/y_train.parquet", f"{self.data_output}/regression/{name}/wt_train.parquet"]],
              "parquet",
              options = {
                "wt_name" : "wt",
              },
              batch_size=self.batch_size,
            )
            regression_means = dp.GetFull(method="mean")
            regression_stds = dp.GetFull(method="std")
            standardisation_parameters["regression"][name] = {}
            for col in regression_means.keys():
              standardisation_parameters["regression"][name][col] = {
                "mean" : regression_means[col],
                "std" : regression_stds[col],
              }

          else:

            for col in sp.keys():
              standardisation_parameters["regression"][name][col] = {
                "mean" : sp[col]["mean"],
                "std" : sp[col]["std"],
              }

    # classifier models
    if self.model_type is None or self.model_type == "classifier_models":
      standardisation_parameters["classifier"] = {}
      for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
        value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
        name = value["parameter"]
        if self.parameter_name is None or self.parameter_name == name:
          sp = GetDictionaryEntry(cfg["preprocess"], ["standardisation",self.file_name,self.category,"classifier",name])

          if sp is None:

            dp = DataProcessor(
              [[f"{self.data_output}/classifier/{name}/X_train.parquet", f"{self.data_output}/classifier/{name}/wt_train.parquet"]],
              "parquet",
              options = {
                "wt_name" : "wt",
              },
              batch_size=self.batch_size,
            )
            classifier_means = dp.GetFull(method="mean")
            classifier_stds = dp.GetFull(method="std")
            standardisation_parameters["classifier"][name] = {}
            for col in classifier_means.keys():
              standardisation_parameters["classifier"][name][col] = {
                "mean" : classifier_means[col],
                "std" : classifier_stds[col],
              }

          else:

            for col in sp.keys():
              standardisation_parameters["classifier"][name][col] = {
                "mean" : sp[col]["mean"],
                "std" : sp[col]["std"],
              }

    return standardisation_parameters


  def _DoStandardisation(self, file_name, cfg, standardisation_parameters):


    for data_split in ["train","test"]:

      if self.model_type is None or self.model_type == "density_models":

        # Check if we need to split density models
        split_density_model = GetSplitDensityModel(cfg, file_name, category=self.category)

        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
          value = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
          if not split_density_model:
            extra_dir = "density"
          else:
            extra_dir = f"density/split_{value['split']}"

          # density model
          dp = DataProcessor(
            [[f"{self.data_output}/{extra_dir}/X_{data_split}.parquet", f"{self.data_output}/{extra_dir}/Y_{data_split}.parquet"]],
            "parquet",
            options = {
              "parameters" : {"standardisation": standardisation_parameters["density"]},
            },
            batch_size=self.batch_size,
          )

          dp.GetFull(
            method=None,
            functions_to_apply = [
              "transform",
              partial(
                self._WriteSplitDataset, 
                extra_dir=f"{extra_dir}", 
                extra_name=f"{data_split}_standardised", 
                split_dict={"X": cfg["variables"], "Y": value["parameters"]}
              )
            ]
          )

          for i in ["X","Y"]:
            os.system(f"mv {self.data_output}/{extra_dir}/{i}_{data_split}_standardised.parquet {self.data_output}/{extra_dir}/{i}_{data_split}.parquet")


      # regression models
      if self.model_type is None or self.model_type == "regression_models":

        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
          value = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
          name = value["parameter"]

          if self.parameter_name is None or self.parameter_name == name:

            dp = DataProcessor(
              [[f"{self.data_output}/regression/{name}/X_{data_split}.parquet", f"{self.data_output}/regression/{name}/y_{data_split}.parquet"]],
              "parquet",
              options = {
                "parameters" : {"standardisation": standardisation_parameters["regression"][name]},
              },
              batch_size=self.batch_size,
            )

            dp.GetFull(
              method=None,
              functions_to_apply = [
                "transform",
                partial(
                  self._WriteSplitDataset, 
                  extra_dir=f"regression/{name}", 
                  extra_name=f"{data_split}_standardised", 
                  split_dict={"X": cfg["variables"]+[value["parameter"]], "y": ["wt_shift"]}
                )
              ]
            )

            for i in ["X","y"]:
              os.system(f"mv {self.data_output}/regression/{name}/{i}_{data_split}_standardised.parquet {self.data_output}/regression/{name}/{i}_{data_split}.parquet")

      # classifier models
      if self.model_type is None or self.model_type == "classifier_models":

        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
          value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
          name = value["parameter"]
          if self.parameter_name is None or self.parameter_name == name:

            dp = DataProcessor(
              [[f"{self.data_output}/classifier/{name}/X_{data_split}.parquet"]],
              "parquet",
              options = {
                "parameters" : {"standardisation": standardisation_parameters["classifier"][name]},
              },
              batch_size=self.batch_size,
            )

            dp.GetFull(
              method=None,
              functions_to_apply = [
                "transform",
                partial(
                  self._WriteSplitDataset, 
                  extra_dir=f"classifier/{name}", 
                  extra_name=f"{data_split}_standardised", 
                  split_dict={"X": cfg["variables"]+[value["parameter"]]}
                )
              ]
            )

            for i in ["X"]:
              os.system(f"mv {self.data_output}/classifier/{name}/{i}_{data_split}_standardised.parquet {self.data_output}/classifier/{name}/{i}_{data_split}.parquet")


  def _DoShuffle(self, file_name, cfg):

    # Check if we need to split density models
    split_density_model = GetSplitDensityModel(cfg, file_name, category=self.category)

    if not split_density_model:
      density_loop = ["density"]
    else:
      split_density_inds = []
      for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
        value = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
        split_density_inds.append(value["split"])
      density_loop = [f"density/split_{ind}" for ind in split_density_inds]

    for data_split in ["train","test"]:

      # density model
      if self.model_type is None or self.model_type == "density_models":
        for extra_dir in density_loop:
          if self.verbose:
            print(f" - Shuffling density model variation for {file_name}, split: {data_split}, directory: {extra_dir}")
          self._DoShuffleDataset([f"{self.data_output}/{extra_dir}/{i}_{data_split}.parquet" for i in ["X","Y","wt","Extra"]])

      # regression models
      if self.model_type is None or self.model_type == "regression_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
          value = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
          name = value["parameter"]
          if self.parameter_name is None or self.parameter_name == name:
            if self.verbose:
              print(f" - Shuffling regression model variation for {file_name}, split: {data_split}, parameter: {name}")
            self._DoShuffleDataset([f"{self.data_output}/regression/{name}/{i}_{data_split}.parquet" for i in ["X","y","wt","Extra"]])

      # classifier models
      if self.model_type is None or self.model_type == "classifier_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
          value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
          name = value["parameter"]
          if self.parameter_name is None or self.parameter_name == name:
            if self.verbose:
              print(f" - Shuffling classifier model variation for {file_name}, split: {data_split}, parameter: {name}")
            self._DoShuffleDataset([f"{self.data_output}/classifier/{name}/{i}_{data_split}.parquet" for i in ["X","y","wt","Extra"]])


  def _DoShuffleDataset(self, files):

    for name in files:

      if not os.path.isfile(name): continue
      shuffle_name_iteration = name.replace(".parquet", "_shuffled_iteration.parquet")
      if os.path.isfile(shuffle_name_iteration):
        os.system(f"rm {shuffle_name_iteration}")   
      shuffle_dp = DataProcessor([[name]],"parquet", batch_size=self.batch_size)

      number_of_shuffles = self.number_of_shuffles

      for shuff in range(number_of_shuffles):
        #print(f" - name={name} shuffle={shuff}")
        shuffle_dp.GetFull(
          method = None,
          functions_to_apply = [
            partial(self._DoShuffleIteration, iteration=shuff, total_iterations=number_of_shuffles, seed=42, dataset_name=shuffle_name_iteration)
          ]
        )
      if os.path.isfile(shuffle_name_iteration):
        os.system(f"mv {shuffle_name_iteration} {name}")

      shuffle_name_batch = name.replace(".parquet", "_shuffled_batch.parquet")
      if os.path.isfile(shuffle_name_batch):
        os.system(f"rm {shuffle_name_batch}")
      shuffle_dp = DataProcessor([[name]],"parquet", batch_size=self.batch_size)
      shuffle_dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(self._DoShuffleBatch, seed=42, dataset_name=shuffle_name_batch)
        ]
      )
      if os.path.isfile(shuffle_name_batch):
        os.system(f"mv {shuffle_name_batch} {name}")

      # Need to shuffle the final batch as otherwise it can be unsorted (as can be smaller than batch size)
      # Move final batch to start
      final_batch_name = name.replace(".parquet", "_final_batch_first.parquet")
      if os.path.isfile(final_batch_name):
        os.system(f"rm {final_batch_name}")
      shuffle_dp = DataProcessor([[name]],"parquet", batch_size=self.batch_size)
      shuffle_dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(self._DoWriteWhetherBatchSize, batch_size=self.batch_size, dataset_name=final_batch_name, do_batch_size=False)
        ]
      )
      shuffle_dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(self._DoWriteWhetherBatchSize, batch_size=self.batch_size, dataset_name=final_batch_name, do_batch_size=True)
        ]
      )

      if os.path.isfile(final_batch_name):
        os.system(f"mv {final_batch_name} {name}")

      # Then shuffle
      shuffle_name_batch_final = name.replace(".parquet", "_shuffled_batch_final.parquet")
      if os.path.isfile(shuffle_name_batch_final):
        os.system(f"rm {shuffle_name_batch_final}")
      shuffle_dp = DataProcessor([[name]],"parquet", batch_size=self.batch_size)
      shuffle_dp.GetFull(
        method = None,
        functions_to_apply = [
          partial(self._DoShuffleBatch, seed=42, dataset_name=shuffle_name_batch_final)
        ]
      )
      if os.path.isfile(shuffle_name_batch_final):
        os.system(f"mv {shuffle_name_batch_final} {name}")


  def _DoWriteWhetherBatchSize(self, df, do_batch_size=False, batch_size=100000, dataset_name="dataset.parquet"):

    if ((len(df) != batch_size) and (not do_batch_size)) or (do_batch_size and (len(df) == batch_size)):

      # Write to file
      table = pa.Table.from_pandas(df, preserve_index=False)
      if os.path.isfile(dataset_name):
        combined_table = pa.concat_tables([pq.read_table(dataset_name), table])
        pq.write_table(combined_table, dataset_name, compression='snappy')
      else:
        pq.write_table(table, dataset_name, compression='snappy')

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


  def _GetParameterExtraName(self):

    extra_name = ""
    if self.partial == "initial":
      extra_name += "_initial"
    elif self.partial == "model":
      extra_name += "_model_type"
      if self.model_type is not None:
        extra_name += f"_{self.model_type}"
        if self.parameter_name is not None:
          extra_name += f"_{self.parameter_name}"
    elif self.partial == "validation":
      extra_name += "_validation"
      if self.val_ind is not None:
        extra_name += f"_val_ind_{self.val_ind}"

    return extra_name


  def _DoParametersFile(self, file_name, cfg, yields={}, eff_events={}, standardisation={}):

    parameters_file = {
      "file_name": file_name, 
      "yields": yields, 
      "density":{}, 
      "regression":{}, 
      "classifier":{},
      "eff_events":eff_events
    }


    parameters_file["density"]["file_loc"] = f"{self.data_output}/density"
    if "density" in standardisation.keys():
      parameters_file["density"]["standardisation"] = standardisation["density"]
    parameters_file["density"]["X_columns"] = cfg["variables"]
    parameters_in_density_model = []
    for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
      v = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
      parameters_in_density_model = sorted(list(set(parameters_in_density_model + v["parameters"])))
    parameters_file["density"]["Y_columns"] = parameters_in_density_model

    # check if we need to split density models
    if GetSplitDensityModel(cfg, file_name, category=self.category):
      parameters_file["density"]["split_Y_columns"] = []
      for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
        v = cfg["models"][file_name]["density_models"][loop_value["loop_index"]]
        parameters_file["density"]["split_Y_columns"].append(v["parameters"])
      parameters_file["density"]["split_Y_columns"] = sorted(list(set(parameters_file["density"]["split_Y_columns"])))

    for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
      v = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
      name = v["parameter"]
      parameters_file["regression"][name] = {}
      parameters_file["regression"][name]["file_loc"] = f"{self.data_output}/regression/{name}"
      if "regression" in standardisation.keys():
        if name in standardisation["regression"].keys():
          parameters_file["regression"][name]["standardisation"] = standardisation["regression"][name]
      parameters_file["regression"][name]["X_columns"] = cfg["variables"] + [v["parameter"]]
      parameters_file["regression"][name]["y_columns"] = ["wt_shift"]

    for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
      v = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
      name = v["parameter"]
      parameters_file["classifier"][name] = {}
      parameters_file["classifier"][name]["file_loc"] = f"{self.data_output}/classifier/{name}"
      if "classifier" in standardisation.keys():
        if name in standardisation["classifier"].keys():
          parameters_file["classifier"][name]["standardisation"] = standardisation["classifier"][name]
      parameters_file["classifier"][name]["X_columns"] = cfg["variables"] + [v["parameter"]]
      parameters_file["classifier"][name]["y_columns"] = ["classifier_truth"]

    if self.binned_fit_input is not None:
      parameters_file["binned_fit_input"] = self.binned_fit_input

    if self.validation_binned is not None:
      parameters_file["validation_binned_fit"] = self.validation_binned

    # write parameters file
    with open(f"{self.data_output}/parameters{self._GetParameterExtraName()}.yaml", 'w') as yaml_file:
      yaml.dump(parameters_file, yaml_file, default_flow_style=False) 


  def _DoFlattenByYields(self, file_name, cfg, yields):

    defaults = GetDefaultsInModel(file_name, cfg, category=self.category)

    yield_class = Yields(
      yields["nominal"],
      lnN = {k:v for k,v in yields["lnN"].items() if k in defaults.keys()},
      physics_model = None,
      rate_param = None,
    )

    def scale_down_by_func(df, func, defaults):
      tmp_df = copy.deepcopy(df)
      for k, v in defaults.items():
        if k not in df.columns:
          tmp_df.loc[:,k] = v
      df.loc[:,"yields"] = func(tmp_df.loc[:,list(defaults.keys())])
      df.loc[:,"wt"] /= df.loc[:,"yields"]
      return df.loc[:,["wt"]]
    
    scale_func = partial(scale_down_by_func, func=yield_class.GetYield, defaults=defaults)

    # Check if we need to split density models
    split_density_model = GetSplitDensityModel(cfg, file_name, category=self.category)

    load_files = []
    edit_files = []
    for data_split in ["train","test"]:

      # density model
      if self.model_type is None or self.model_type == "density_models":
        if not split_density_model:
          load_files.append([f"{self.data_output}/density/X_{data_split}.parquet", f"{self.data_output}/density/Y_{data_split}.parquet", f"{self.data_output}/density/wt_{data_split}.parquet"])
          edit_files.append(f"density/wt_{data_split}.parquet")
        else:
          for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
            ind = loop_value["split"]
            load_files.append([f"{self.data_output}/density/split_{ind}/X_{data_split}.parquet", f"{self.data_output}/density/split_{ind}/Y_{data_split}.parquet", f"{self.data_output}/density/split_{ind}/wt_{data_split}.parquet"])
            edit_files.append(f"density/split_{ind}/wt_{data_split}.parquet")

      # regression models
      if self.model_type is None or self.model_type == "regression_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
          value = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
          name = value["parameter"]
          if self.parameter_name is None or self.parameter_name == name:
            load_files.append([f"{self.data_output}/regression/{name}/X_{data_split}.parquet", f"{self.data_output}/regression/{name}/y_{data_split}.parquet", f"{self.data_output}/regression/{name}/wt_{data_split}.parquet"])
            edit_files.append(f"regression/{name}/wt_{data_split}.parquet")

      # classifier models
      if self.model_type is None or self.model_type == "classifier_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
          value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
          name = value["parameter"]
          if self.parameter_name is None or self.parameter_name == name:
            load_files.append([f"{self.data_output}/classifier/{name}/X_{data_split}.parquet", f"{self.data_output}/classifier/{name}/y_{data_split}.parquet", f"{self.data_output}/classifier/{name}/wt_{data_split}.parquet"])
            edit_files.append(f"classifier/{name}/wt_{data_split}.parquet")

    for file_ind, file_names in enumerate(load_files):

      if not os.path.isfile(f"{self.data_output}/{edit_files[file_ind]}"): continue
      flattened_name = edit_files[file_ind].replace(".parquet", "_flattened.parquet")
      if os.path.isfile(f"{self.data_output}/{flattened_name}"):
        os.system(f"rm {self.data_output}/{flattened_name}")   

      flat_dp = DataProcessor([file_names],"parquet", batch_size=self.batch_size)
      flat_dp.GetFull(
        method = None,
        functions_to_apply = [
          scale_func,
          partial(self._WriteDataset, file_name=flattened_name)
        ]
      )

      if os.path.isfile(f"{self.data_output}/{flattened_name}"):
        os.system(f"mv {self.data_output}/{flattened_name} {self.data_output}/{edit_files[file_ind]}")  


  def _DoNormaliseIn(self, file_name, cfg):

    if "normalise_training_in" not in cfg["preprocess"]:
      return None
    if file_name not in cfg["preprocess"]["normalise_training_in"]:
      return None
    if len(cfg["preprocess"]["normalise_training_in"][file_name]) == 0:
      return None

    # Check if we need to split density models
    split_density_model = GetSplitDensityModel(cfg, file_name, category=self.category)

    load_files = {}
    edit_files = {}
    for data_split in ["train","test"]:
      load_files[data_split] = []
      edit_files[data_split] = []

      # density model
      if self.model_type is None or self.model_type == "density_models":
        if not split_density_model:
          load_files[data_split].append([f"{self.data_output}/density/X_{data_split}.parquet", f"{self.data_output}/density/Y_{data_split}.parquet", f"{self.data_output}/density/wt_{data_split}.parquet"])
          edit_files[data_split].append(f"density/wt_{data_split}.parquet")
        else:
          for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=self.category):
            ind = loop_value["split"]
            load_files[data_split].append([f"{self.data_output}/density/split_{ind}/X_{data_split}.parquet", f"{self.data_output}/density/split_{ind}/Y_{data_split}.parquet", f"{self.data_output}/density/split_{ind}/wt_{data_split}.parquet"])
            edit_files[data_split].append(f"density/split_{ind}/wt_{data_split}.parquet")

      # regression models
      if self.model_type is None or self.model_type == "regression_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=self.category):
          value = cfg["models"][file_name]["regression_models"][loop_value["loop_index"]]
          name = value["parameter"]
          if self.parameter_name is None or self.parameter_name == name:
            load_files[data_split].append([f"{self.data_output}/regression/{name}/X_{data_split}.parquet", f"{self.data_output}/regression/{name}/y_{data_split}.parquet", f"{self.data_output}/regression/{name}/wt_{data_split}.parquet"])
            edit_files[data_split].append(f"regression/{name}/wt_{data_split}.parquet")

      # classifier models
      if self.model_type is None or self.model_type == "classifier_models":
        for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
          value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]
          name = value["parameter"]
          if self.parameter_name is None or self.parameter_name == name:
            load_files[data_split].append([f"{self.data_output}/classifier/{name}/X_{data_split}.parquet", f"{self.data_output}/classifier/{name}/y_{data_split}.parquet", f"{self.data_output}/classifier/{name}/wt_{data_split}.parquet"])
            edit_files[data_split].append(f"classifier/{name}/wt_{data_split}.parquet")

    # get normalisations
    normalisations = {}
    for file_ind, file_names in enumerate(load_files["train"]):
      normalised_dp = DataProcessor(
        [file_names], 
        "parquet", 
        batch_size=self.batch_size,
        options = {
          "wt_name" : "wt",
        }
      )
      normalisations[file_ind] = []
      for sel in cfg["preprocess"]["normalise_training_in"][file_name]:
        normalisations[file_ind].append(
          normalised_dp.GetFull(method="sum", extra_sel=sel)
        )

    for data_split in ["train","test"]:

      for file_ind, file_names in enumerate(load_files[data_split]):

        if not os.path.isfile(f"{self.data_output}/{edit_files[data_split][file_ind]}"): continue
        normalised_name = edit_files[data_split][file_ind].replace(".parquet", "_normalised.parquet")
        if os.path.isfile(f"{self.data_output}/{normalised_name}"):
          os.system(f"rm {self.data_output}/{normalised_name}")   

        normalised_dp = DataProcessor(
          [file_names], 
          "parquet", 
          batch_size=self.batch_size,
          options = {
            "wt_name" : "wt",
          }
        )

        def normalisation_func(df, normalisations=[], selections=[]):
          if len(df) == 0: 
            return df
          for ind, sel in enumerate(selections):
            df.loc[(df.eval(sel)),"wt"] /= normalisations[ind]
          return df.loc[:,["wt"]]


        normalised_dp.GetFull(
          method = None,
          functions_to_apply = [
            partial(normalisation_func, normalisations=normalisations[file_ind], selections=cfg["preprocess"]["normalise_training_in"][file_name]),
            partial(self._WriteDataset, file_name=normalised_name)
          ]
        )

        if os.path.isfile(f"{self.data_output}/{normalised_name}"):
          os.system(f"mv {self.data_output}/{normalised_name} {self.data_output}/{edit_files[data_split][file_ind]}")  



  def _DoClassBalancing(self, file_name, cfg):

    if self.model_type is None or self.model_type == "classifier_models":

      for loop_value in GetModelLoop(cfg, model_file_name=file_name, only_classification=True, specific_category=self.category):
        value = cfg["models"][file_name]["classifier_models"][loop_value["loop_index"]]

        if self.parameter_name is None or self.parameter_name == value["parameter"]:

          dp = DataProcessor(
            [[f"{self.data_output}/classifier/{value['parameter']}/y_train.parquet", f"{self.data_output}/classifier/{value['parameter']}/wt_train.parquet"]],
            "parquet",
            options = {
              "wt_name" : "wt",
              "selection" : None,
            },
            batch_size=self.batch_size,
          )
          sum_zero = dp.GetFull(method="sum", extra_sel="(classifier_truth == 0)")
          sum_one = dp.GetFull(method="sum", extra_sel="(classifier_truth == 1)")
          scale_to = max(sum_zero, sum_one)

          def class_balancing(df):
            df.loc[(df["classifier_truth"] == 0), "wt"] *= scale_to / sum_zero
            df.loc[(df["classifier_truth"] == 1), "wt"] *= scale_to / sum_one
            return df.loc[:,["wt"]]

          for tt in ["train","test"]:

            balanced_name = f"classifier/{value['parameter']}/wt_{tt}_balanced.parquet"
            if os.path.isfile(f"{self.data_output}/{balanced_name}"):
              os.system(f"rm {self.data_output}/{balanced_name}")

            wdp = DataProcessor(
              [[f"{self.data_output}/classifier/{value['parameter']}/y_{tt}.parquet", f"{self.data_output}/classifier/{value['parameter']}/wt_{tt}.parquet"]],
              "parquet",
              options = {
                "wt_name" : "wt",
                "selection" : None,
              },
              batch_size=self.batch_size,
            )

            wdp.GetFull(
              method = None,
              functions_to_apply = [
                class_balancing,
                partial(self._WriteDataset, file_name=balanced_name)
              ]
            )
            if os.path.isfile(f"{self.data_output}/{balanced_name}"):
              os.system(f"mv {self.data_output}/{balanced_name} {self.data_output}/classifier/{value['parameter']}/wt_{tt}.parquet")


  def _DoMergeParametersFile(self, file_name, cfg):

    first = True
    for parameter_extra_name in self.merge_parameters:
      
      if parameter_extra_name is None:
        en = ""
      else:
        en = f"_{parameter_extra_name}"

      # load initial parameters file
      with open(f"{self.data_output}/parameters{en}.yaml", 'r') as yaml_file:
        temp_parameters_file = yaml.safe_load(yaml_file)
      
      if first:
        parameters_file = copy.deepcopy(temp_parameters_file)
        first = False
      else:
        tmp_keys, tmp_vals = FindKeysAndValuesInDictionaries(copy.deepcopy(temp_parameters_file))
        for key, val in zip(tmp_keys, tmp_vals):
          if val is None: continue
          if GetDictionaryEntry(parameters_file, key) is None:
            parameters_file = MakeDictionaryEntry(copy.deepcopy(parameters_file), key, val)

    # write parameters file
    with open(f"{self.data_output}/parameters.yaml", 'w') as yaml_file:
      yaml.dump(parameters_file, yaml_file, default_flow_style=False) 


  def _DoNuisanceVariations(self, file_name, cfg, do_clear=True):
    
    if len(cfg["nuisances"]) == 0:
      return
    
    # Get parameters in model
    parameters_in_model = GetParametersInModel(file_name, cfg, category=self.category)

    # Get extra columns
    extra_cols = None
    if "save_extra_columns" in cfg["preprocess"]:
      if file_name in cfg["preprocess"]["save_extra_columns"]:
        extra_cols = cfg["preprocess"]["save_extra_columns"][file_name]

    # Get defaults
    defaults = GetDefaults(cfg)
    default_index = GetValidationDefaultIndex(cfg, file_name)

    for nui in cfg["nuisances"]:

      if nui not in parameters_in_model:
        continue

      for shift in ["up","down"]:
        
        val_key = f"{nui}_{shift}"

        if self.nuisance_shift is None or self.nuisance_shift == val_key:

          # Get base file from validation
          base_file_name = None
          for value in cfg["validation"]["files"][file_name]:
            if "categories" not in value.keys() or self.category is None or self.category in value["categories"]:
              base_file_name = value["file"]
              break

          # Setup shift dictionary
          parameters_in_file = cfg["files"][base_file_name]["parameters"]
          shifts = {}
          for k, v in defaults.items():
            if k not in parameters_in_file:
              shifts[k] = {"type":"fixed","value":v}
          shifts[nui] = {"type":"fixed", "value": 1.0 if shift=="up" else -1.0}
          val_shift = {
            "parameters" : parameters_in_model,
            "file" : base_file_name,
            "n_copies" : 1,
            "shifts" : shifts
          }

          # Loop through validation types
          for data_split in ["val","train_inf","test_inf","full"]:
            
            # Clear previous files
            for k in ["X","Y","wt","Extra"]:
              outfile = f"{self.data_output}/{val_key}/{k}_{data_split}.parquet"
              if os.path.isfile(outfile):
                os.system(f"rm {outfile}")      

            # Do variations            
            val_split_model = {"X":cfg["variables"], "Y":val_shift["parameters"], "wt":["wt"]}
            if extra_cols is not None:
              val_split_model["Extra"] = extra_cols 
            self._DoWriteModelVariation(val_shift, self.data_output, f"val_ind_{default_index}/{base_file_name}_{data_split}", cfg, f"{val_key}", data_split, split_dict=val_split_model)

    if do_clear:
      # Clear up old files
      for k in cfg["files"].keys():
        for data_split in ["val","train_inf","test_inf","full"]:
          for ind in range(len(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True))):
            if ind != default_index: continue
            outfile = f"{self.data_output}/val_ind_{ind}/{k}_{data_split}.parquet"
            if os.path.isfile(outfile):
              os.system(f"rm {outfile}")  


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

    # Set parameters needed
    eff_events = {}
    standardisation_parameters = {}

    # Load config
    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg) 
  

    # Do initial methods
    if self.partial is None or self.partial == "initial":

      # Make yields
      if self.verbose:
        print("- Calculating yields")
      yields = self._GetYields(self.file_name, cfg)

      # Get binned fit inputs
      if self.verbose:
        print("- Get binned fit inputs")
      self._GetBinnedFitInputs(self.file_name, cfg, yields)

      # Train/Val split the dataset
      if self.verbose:
        print("- Doing train/test/val splitting")
      self._DoTrainTestValSplit(self.file_name, cfg)


    # Load yields
    if self.partial is not None and self.partial in ["model", "validation", "nuisance_variations"]:
      if self.verbose:
        print("- Loading in yields")
      # load initial parameters file
      with open(f"{self.data_output}/parameters_initial.yaml", 'r') as yaml_file:
        parameters_file = yaml.safe_load(yaml_file)
      yields = parameters_file["yields"]


    # Make dataset for model training and testing
    if self.partial is None or self.partial == "model":

      # Do variations in each of the train/test/test_inf/val datasets
      if self.verbose:
        print("- Calculating variations for each dataset")
      self._DoModelVariations(self.file_name, cfg, do_clear=(self.partial is None))

      # Flatten train/test with from yields
      if self.verbose:
        print("- Flattening train and test dataset")
      self._DoFlattenByYields(self.file_name, cfg, yields)

      # Do shift reweighting
      if self.verbose:
        print("- Reweighting shifts in train and test datasets back to chosen distributions")
      self._DoShiftReweighting(self.file_name, cfg)

      # Normalise yields in certain bins
      if self.verbose:
        print("- Normalising train and test in given selections")
      self._DoNormaliseIn(self.file_name, cfg)

      # Standardise
      if self.verbose:
        print("- Standardising train and test datasets")
      standardisation_parameters = self._GetStandardisationParameters(self.file_name, cfg)
      self._DoStandardisation(self.file_name, cfg, standardisation_parameters)

      # Do classifier class balancing
      if self.verbose:
        print("- Doing classifier class balancing")
      self._DoClassBalancing(self.file_name, cfg)

      # Shuffle
      if self.verbose:
        print("- Shuffling train and test datasets")
      self._DoShuffle(self.file_name, cfg)


    # Make validation datasets
    if self.partial is None or self.partial == "validation":

      # Do the validation variations
      if self.verbose:
        print("- Calculating variations for validation datasets")
      self._DoValidationVariations(self.file_name, cfg, do_clear=(self.partial is None))

      # Normalise validations datasets to correct yield
      if self.verbose:
        print("- Normalising validation datasets to correct yield")
      self._DoValidationNormalisation(self.file_name, cfg, yields)

      # Get effective events of validation datasets
      if self.verbose:
        print("- Getting effective events in each validation dataset")
      eff_events = self._GetValidationEffEvents(self.file_name, cfg)

      # Get validation binned fit inputs
      if self.verbose:
        print("- Get validation binned fit inputs")
      self._GetValidationBinned(self.file_name, cfg)


    # Make nuisance variation samples
    if self.partial is None or self.partial == "nuisance_variations":
      
      # Make nuisance variation datasets
      if self.verbose:
        print("- Calculating nuisance variation datasets")
      self._DoNuisanceVariations(self.file_name, cfg, do_clear=(self.partial is None))

      # Normalise nuisance variation datasets to correct yield
      if self.verbose:
        print("- Normalising nuisance variation datasets to correct yield")
      self._DoNuisanceNormalisation(self.file_name, cfg, yields)


    # Write parameters
    if self.partial is None or self.partial not in ["merge","nuisance_variations"]:
      if self.verbose:
        print("- Writing parameters")
      self._DoParametersFile(self.file_name, cfg, yields=yields, eff_events=eff_events, standardisation=standardisation_parameters)


    # Merge parameters
    if self.partial == "merge":
      if self.verbose:
        print("- Merging parameters")
      self._DoMergeParametersFile(self.file_name, cfg)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add parameters file
    if self.partial is None or self.partial in ["initial", "model", "validation","merge"]:
      outputs += [f"{self.data_output}/parameters{self._GetParameterExtraName()}.yaml"]
      
    # Check if we output Extra
    density_loop = ["X","Y","wt"]
    regression_loop = ["X","y","wt"]
    classifier_loop = ["X","y","wt"]
    if "save_extra_columns" in cfg["preprocess"]:
      if self.file_name in cfg["preprocess"]["save_extra_columns"]:
        density_loop += ["Extra"]
        regression_loop += ["Extra"]
        classifier_loop += ["Extra"]


    # Do models
    if self.partial is None or self.partial == "model":

      # Add train/test files
      for data_split in ["train","test"]:

        # Add density files
        if self.model_type is None or self.model_type == "density_models":
          for k in density_loop:
            if not GetSplitDensityModel(cfg, self.file_name, category=self.category):
              outputs += [f"{self.data_output}/density/{k}_{data_split}.parquet"]
            else:
              for loop_value in GetValidationLoop(cfg, self.file_name, include_rate=True, include_lnN=True):
                ind = loop_value["split"]
                outputs += [f"{self.data_output}/density/split_{ind}/{k}_{data_split}.parquet"]

        # Add regression files
        if self.model_type is None or self.model_type == "regression_models":
          for k in regression_loop:
            for loop_value in GetModelLoop(cfg, self.file_name, only_regression=True, specific_category=self.category):
              value = cfg["models"][self.file_name]["regression_models"][loop_value["loop_index"]]
              if self.parameter_name is None or self.parameter_name == value["parameter"]:
                outputs += [f"{self.data_output}/regression/{value['parameter']}/{k}_{data_split}.parquet"]

        # Add classifier files
        if self.model_type is None or self.model_type == "classifier_models":
          for k in classifier_loop:
            for loop_value in GetModelLoop(cfg, self.file_name, only_classification=True, specific_category=self.category):
              value = cfg["models"][self.file_name]["classifier_models"][loop_value["loop_index"]]
              if self.parameter_name is None or self.parameter_name == value["parameter"]:
                outputs += [f"{self.data_output}/classifier/{value['parameter']}/{k}_{data_split}.parquet"]

    # Do validation
    if self.partial is None or self.partial == "validation":

      # Add validation files
      for data_split in ["val","train_inf","test_inf","full"]:
        for k in density_loop:
          if self.partial is None:
            val_ind_loop = range(len(GetValidationLoop(cfg, self.file_name, include_rate=True, include_lnN=True)))
          else:
            val_ind_loop = [self.val_ind]
          for ind in val_ind_loop:
            outputs += [f"{self.data_output}/val_ind_{ind}/{k}_{data_split}.parquet"]

    # Do nuisance variations
    if self.partial is None or self.partial == "nuisance_variations":
      
      for nui in cfg["nuisances"]:
        parameters_in_model = GetParametersInModel(self.file_name, cfg, category=self.category)
        if nui not in parameters_in_model: continue
        for shift in ["up","down"]:
          val_key = f"{nui}_{shift}"
          if self.nuisance_shift is None or self.nuisance_shift == val_key:
            for data_split in ["val","train_inf","test_inf","full"]:
              for k in density_loop:
                outputs += [f"{self.data_output}/{val_key}/{k}_{data_split}.parquet"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Add config
    inputs = [self.cfg]

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add yields base files
    if "yields" in cfg["models"][self.file_name]:
      base_file_name = GetYieldsBaseFile(cfg["models"][self.file_name]["yields"], self.category)
      inputs += [f"{self.data_input}/{base_file_name}.parquet"]

    # Add validation base files
    val_base_file_name = GetYieldsBaseFile(cfg["validation"]["files"][self.file_name], self.category)
    inputs += [f"{self.data_input}/{val_base_file_name}.parquet"]

    # Add variation base files
    files_in_model = GetFilesInModel(self.file_name, cfg)
    for uf in files_in_model:
      file_name = f"{self.data_input}/{uf}.parquet"
      if file_name not in inputs:
        inputs += [file_name]

    # Add initial parameters
    if self.partial in ["model", "validation", "nuisance_variations"]:
      inputs += [f"{self.data_output}/parameters_initial.yaml"]

    # Add merge parameters
    if self.partial == "merge":
      for parameter_extra_name in self.merge_parameters:
        if parameter_extra_name is None:
          en = ""
        else:
          en = f"_{parameter_extra_name}"
        inputs += [f"{self.data_output}/parameters{en}.yaml"]

    return inputs
