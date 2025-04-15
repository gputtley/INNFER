import copy
import os
import warnings
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor
from useful_functions import (
    BuildBinnedCategories,
    GetDefaults,
    GetDefaultsInModel,
    GetFilesInModel,
    GetParametersInModel,
    GetValidationLoop,
    LoadConfig,
    MakeDirectories
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

    # Other
    self.number_of_shuffles = 10
    self.verbose = True
    self.data_input = "data/"
    self.data_output = "data/"
    self.seed = 2
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_PREPROCESS"))
    self.binned_fit_input = None

    # Stores
    self.parameters = {}


  def _CalculateDatasetWithVariations(self, df, shifts, pre_calculate, post_calculate_selection, weight_shifts, nominal_weight, n_copies=1):

    # Distribute the shifts
    tmp_copy = copy.deepcopy(df)
    first = True
    for _ in range(n_copies):
      tmp = copy.deepcopy(tmp_copy)
      for k, v in shifts.items():
        if v["type"] == "continuous":
          tmp.loc[:,k] = np.random.uniform(v["range"][0], v["range"][1], size=len(tmp))   
        elif v["type"] == "discrete":
          tmp.loc[:,k] = np.random.choice(v["values"], size=len(tmp))
        elif v["type"] == "fixed":
          tmp.loc[:,k] = v["value"]*np.ones(len(tmp))
      if first:
        df = copy.deepcopy(tmp)
        first = False
      else:
        df = pd.concat([df, tmp], axis=0, ignore_index=True)

    # do precalculate
    for pre_calc_col_name, pre_calc_col_value in pre_calculate.items():
      df.loc[:,pre_calc_col_name] = df.eval(pre_calc_col_value)          

    # Apply post selection
    if post_calculate_selection is not None:
      df = df.loc[df.eval(post_calculate_selection),:]

    # store old weight before doing weight shift
    df.loc[:,"old_wt"] = df.eval(nominal_weight)

    # do weight shift
    weight_shift_function = nominal_weight
    for k, v in weight_shifts.items():
      weight_shift_function += f"*({v})"

    df.loc[:,"wt"] = df.eval(weight_shift_function)
    df.loc[:,"wt_shift"] = df.loc[:,"wt"]/df.loc[:,"old_wt"]

    return df


  def _GetYields(self, file_name, cfg, sigma=1.0, extra_sel=None):

    # initiate dictionary
    yields = {}

    # Get defaults
    defaults = GetDefaults(cfg)

    # Defaults that are not in the dataset
    base_file_name = cfg["models"][file_name]["yields"]["file"]
    parameters_of_file = cfg["files"][base_file_name]["parameters"]
    nominal_shifts = {k: {"type":"fixed","value":v} for k,v in defaults.items() if k not in parameters_of_file}
    if len(parameters_of_file) > 0:
      selection = " & ".join([f"({k}=={defaults[k]})" for k in parameters_of_file])
    else:
      selection = None

    # Add extra selection
    if extra_sel is not None:
      if selection is None:
        selection = extra_sel
      else:
        selection = f"(({extra_sel}) & ({selection}))"

    # Build dataprocessor
    dp = DataProcessor(
      [f"{self.data_input}/{cfg['models'][file_name]['yields']['file']}.parquet"],
      "parquet",
      options = {
        "wt_name" : "wt",
      },
      batch_size=self.batch_size,
    )

    # Get nominal
    yields["nominal"] = dp.GetFull(
      method="sum",
      functions_to_apply = [
        partial(
          self._CalculateDatasetWithVariations, 
          shifts=nominal_shifts, 
          pre_calculate=cfg["files"][base_file_name]["pre_calculate"], 
          post_calculate_selection=cfg["files"][base_file_name]["post_calculate_selection"], 
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


        up_yield = dp.GetFull(
          method="sum",
          functions_to_apply = [
            partial(
              self._CalculateDatasetWithVariations, 
              shifts=up_shifts, 
              pre_calculate=cfg["files"][base_file_name]["pre_calculate"], 
              post_calculate_selection=cfg["files"][base_file_name]["post_calculate_selection"], 
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
              post_calculate_selection=cfg["files"][base_file_name]["post_calculate_selection"], 
              weight_shifts=cfg["files"][base_file_name]["weight_shifts"],
              n_copies=1,
              nominal_weight=cfg["files"][base_file_name]["weight"],
            )
          ],
          extra_sel = down_extra_sel,
        )

        yields["lnN"][nui] = [down_yield/yields["nominal"], up_yield/yields["nominal"]]

    # Add log normals that are in the models
    for key, val in cfg["inference"]["lnN"].items():
      if file_name in val["files"]:
        if isinstance(val["rate"],list):
          yields["lnN"][key] = val["rate"]
        else:
          yields["lnN"][key] = [1/val["rate"], val["rate"]]

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

    # check which validation file need to be split
    files_in_model = GetFilesInModel(file_name, cfg)
    if cfg["validation"]["files"][file_name] in files_in_model:
      do_split = True
    else:
      do_split = False

    # get validation loop and their selections
    validation_loop = GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True)
    parameters_in_file = cfg["files"][cfg["validation"]["files"][file_name]]["parameters"]
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
      outfile = f"{self.data_output}/{cfg['validation']['files'][file_name]}_{i}.parquet"
      if os.path.isfile(outfile):
        os.system(f"rm {outfile}")
    for i in ["train_inf","test_inf", "val","full"]:
      for ind in range(len(selections)):
        outfile = f"{self.data_output}/val_ind_{ind}/{cfg['validation']['files'][file_name]}_{i}.parquet"
        if os.path.isfile(outfile):
          os.system(f"rm {outfile}")

    # Run train test validation on common file split
    dp = DataProcessor(
      [f"{self.data_input}/{cfg['validation']['files'][file_name]}.parquet"],
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
          file_name=cfg['validation']['files'][file_name], 
          splitting=cfg["preprocess"]["train_test_val_split"], 
          validation_only=(not do_split), 
          drop_for_training_selection=unused_training_selection, 
          validation_loop_selections=selections
        )
      ]
    )

    # Run train test split on non validation files
    unique_files = copy.deepcopy(files_in_model)
    if cfg["validation"]["files"][file_name] in unique_files:
      unique_files.remove(cfg["validation"]["files"][file_name])

    for uf in unique_files:

      # Delete files
      for i in ["train","test"]:
        outfile = f"{self.data_output}/{uf}_{i}.parquet"
        if os.path.isfile(outfile):
          os.system(f"rm {outfile}")

      # Run train test validation on common file split
      dp = DataProcessor(
        [f"{self.data_input}/{uf}.parquet"],
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
            file_name=uf, 
            splitting=cfg["preprocess"]["train_test_val_split"], 
            validation_only=False,
            train_test_only=True,
            drop_for_training_selection=unused_training_selection, 
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

      print(self.batch_size, value["n_copies"])
      print(int(np.ceil(self.batch_size/value["n_copies"])))

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
        columns = list(set(columns))

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
            ),
            partial(
              self._WriteSplitDataset, 
              extra_dir=extra_dir, 
              extra_name=extra_name, 
              split_dict=split_dict
            )
          ]
        )


  def _DoModelVariations(self, file_name, cfg):

    # Split models
    parameters_in_density_model = []
    for v in cfg["models"][file_name]["density_models"]:
      parameters_in_density_model = list(set(parameters_in_density_model + v["parameters"]))
    density_split_model = {"X":cfg["variables"],"Y":parameters_in_density_model,"wt":["wt"]}

    # Get extra columns
    extra_cols = None
    if "save_extra_columns" in cfg["preprocess"]:
      if file_name in cfg["preprocess"]["save_extra_columns"]:
        extra_cols = cfg["preprocess"]["save_extra_columns"][file_name]
        density_split_model["Extra"] = extra_cols

    # Delete files
    for data_split in ["train","test"]:
      for k in ["X","Y","wt","Extra"]:
        outfile = f"{self.data_output}/density/{k}_{data_split}.parquet"
        if os.path.isfile(outfile):
          os.system(f"rm {outfile}")
        for k in ["X","y","wt","Extra"]:
          for value in cfg["models"][file_name]["regression_models"]:
            outfile = f"{self.data_output}/regression/{value['parameter']}/{k}_{data_split}.parquet"
            if os.path.isfile(outfile):
              os.system(f"rm {outfile}")
    for data_split in ["val","train_inf","test_inf","full"]:
      for k in ["X","Y","wt","Extra"]:
        for ind in range(len(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True))):
          outfile = f"{self.data_output}/val_ind_{ind}/{k}_{data_split}.parquet"
          if os.path.isfile(outfile):
            os.system(f"rm {outfile}")      

    # Get defaults
    defaults = GetDefaults(cfg)

    # Loop through the train and test
    for data_split in ["train","test"]:
      for value in cfg["models"][file_name]["density_models"]:
        value_copy = copy.deepcopy(value)
        for k, v in defaults.items():
          if k not in value["parameters"]:
            value_copy["shifts"][k] = {"type":"fixed","value":v}
        self._DoWriteModelVariation(value_copy, self.data_output, f"{value['file']}_{data_split}", cfg, "density", data_split, split_dict=density_split_model)
      for value in cfg["models"][file_name]["regression_models"]:
        value_copy = copy.deepcopy(value)
        for k, v in defaults.items():
          if k != value["parameter"]:
            value_copy["shifts"][k] = {"type":"fixed","value":v}
        regression_split_model = {"X":cfg["variables"]+[value["parameter"]], "y":["wt_shift"], "wt":["old_wt"]}
        if extra_cols is not None:
          regression_split_model["Extra"] = extra_cols
        self._DoWriteModelVariation(value_copy, self.data_output, f"{value['file']}_{data_split}", cfg, f"regression/{value['parameter']}", data_split, split_dict=regression_split_model)

    # Loop through validation files
    for data_split in ["val","train_inf","test_inf","full"]:
      for ind, val in enumerate(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True)):

        base_file_name = cfg["validation"]["files"][file_name]
        parameters_in_file = cfg["files"][base_file_name]["parameters"]
        shift_parameters = [k for k in val.keys() if k not in parameters_in_file]
        val_shift = {
          "parameters" : GetParametersInModel(file_name, cfg),
          "file" : base_file_name,
          "n_copies" : 1,
          "shifts" : {k: {"type":"fixed","value":val[k]} for k in shift_parameters}
        }
        val_split_model = {"X":cfg["variables"], "Y":val_shift["parameters"], "wt":["wt"]}
        if extra_cols is not None:
          val_split_model["Extra"] = extra_cols 
        self._DoWriteModelVariation(val_shift, self.data_output, f"val_ind_{ind}/{base_file_name}_{data_split}", cfg, f"val_ind_{ind}", data_split, split_dict=val_split_model)

    # Clear up old files
    for k in cfg["files"].keys():
      for data_split in ["train","test"]:
        outfile = f"{self.data_output}/{k}_{data_split}.parquet"
        if os.path.isfile(outfile):
          os.system(f"rm {outfile}")
      for data_split in ["val","train_inf","test_inf","full"]:
        for ind in range(len(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True))):
          outfile = f"{self.data_output}/val_ind_{ind}/{k}_{data_split}.parquet"
          if os.path.isfile(outfile):
            os.system(f"rm {outfile}")    


  def _DoValidationNormalisation(self, file_name, cfg, yields):

    for data_split in ["val","train_inf","test_inf","full"]:

      for ind, val_loop in enumerate(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True)):

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

        params_in_model = GetDefaultsInModel(file_name, cfg, include_rate=True, include_lnN=True)
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
    

  def _GetBinnedFitInputs(self, file_name, cfg):

    categories = BuildBinnedCategories(self.binned_fit_input)
    print(categories)
    exit()


  def _GetValidationEffEvents(self, file_name, cfg):

    eff_events = {}
    for data_split in ["val","train_inf","test_inf","full"]:
      eff_events[data_split] = {}
      for ind in range(len(GetValidationLoop(cfg, file_name, include_rate=True, include_lnN=True))):

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

    # density model
    dp = DataProcessor(
      [[f"{self.data_output}/density/X_train.parquet", f"{self.data_output}/density/Y_train.parquet", f"{self.data_output}/density/wt_train.parquet"]],
      "parquet",
      options = {
        "wt_name" : "wt",
      },
      batch_size=self.batch_size,
    )
    density_means = dp.GetFull(method="mean")
    density_stds = dp.GetFull(method="std")
    standardisation_parameters["density"] = {}
    for col in density_means.keys():
      standardisation_parameters["density"][col] = {
        "mean" : density_means[col],
        "std" : density_stds[col],
      }

    # regression models
    standardisation_parameters["regression"] = {}
    for value in cfg["models"][file_name]["regression_models"]:
      name = value["parameter"]

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

    return standardisation_parameters

  def _DoStandardisation(self, file_name, cfg, standardisation_parameters):

    for data_split in ["train","test"]:

      # density model
      dp = DataProcessor(
        [[f"{self.data_output}/density/X_{data_split}.parquet", f"{self.data_output}/density/Y_{data_split}.parquet"]],
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
            extra_dir="density", 
            extra_name=f"{data_split}_standardised", 
            split_dict={"X": cfg["variables"], "Y": [k for k in standardisation_parameters["density"].keys() if k not in cfg["variables"]]}
          )
        ]
      )

      for i in ["X","Y"]:
        os.system(f"mv {self.data_output}/density/{i}_{data_split}_standardised.parquet {self.data_output}/density/{i}_{data_split}.parquet")


      # regression models
      for value in cfg["models"][file_name]["regression_models"]:
        name = value["parameter"]

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


  def _DoShuffle(self, file_name, cfg):

    for data_split in ["train","test"]:

      # density model
      self._DoShuffleDataset([f"{self.data_output}/density/{i}_{data_split}.parquet" for i in ["X","Y","wt","Extra"]])

      # regression models
      for value in cfg["models"][file_name]["regression_models"]:
        name = value["parameter"]
        self._DoShuffleDataset([f"{self.data_output}/regression/{name}/{i}_{data_split}.parquet" for i in ["X","y","wt","Extra"]])

  def _DoShuffleDataset(self, files):

    for name in files:
      if not os.path.isfile(name): continue
      shuffle_name = name.replace(".parquet", "_shuffled.parquet")
      if os.path.isfile(shuffle_name):
        os.system(f"rm {shuffle_name}")   
      shuffle_dp = DataProcessor([[name]],"parquet", batch_size=self.batch_size)
      for shuff in range(self.number_of_shuffles):
        #print(f" - name={name} shuffle={shuff}")
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


  def _DoParametersFile(self, file_name, cfg, yields={}, eff_events={}, standardisation={}):

    parameters_file = {
      "file_name": file_name, 
      "yields": yields, 
      "density":{}, 
      "regression":{}, 
      "eff_events":eff_events
    }

    parameters_file["density"]["file_loc"] = f"{self.data_output}/density"
    if "density" in standardisation.keys():
      parameters_file["density"]["standardisation"] = standardisation["density"]
    parameters_file["density"]["X_columns"] = cfg["variables"]
    parameters_in_density_model = []
    for v in cfg["models"][file_name]["density_models"]:
      parameters_in_density_model = list(set(parameters_in_density_model + v["parameters"]))
    parameters_file["density"]["Y_columns"] = parameters_in_density_model

    for v in cfg["models"][file_name]["regression_models"]:
      name = v["parameter"]
      parameters_file["regression"][name] = {}
      parameters_file["regression"][name]["file_loc"] = f"{self.data_output}/regression/{name}"
      if "regression" in standardisation.keys():
        if name in standardisation["regression"].keys():
          parameters_file["regression"][name]["standardisation"] = standardisation["regression"][name]
      parameters_file["regression"][name]["X_columns"] = cfg["variables"] + [v["parameter"]]
      parameters_file["regression"][name]["y_columns"] = ["wt_shift"]

    # write parameters file
    with open(self.data_output+"/parameters.yaml", 'w') as yaml_file:
      yaml.dump(parameters_file, yaml_file, default_flow_style=False) 


  def _DoFlattenByYields(self, file_name, cfg, yields):

    defaults = GetDefaultsInModel(file_name, cfg)

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

    load_files = []
    edit_files = []
    for data_split in ["train","test"]:
      # density model
      load_files.append([f"{self.data_output}/density/X_{data_split}.parquet", f"{self.data_output}/density/Y_{data_split}.parquet", f"{self.data_output}/density/wt_{data_split}.parquet"])
      edit_files.append(f"density/wt_{data_split}.parquet")
      # regression models
      for value in cfg["models"][file_name]["regression_models"]:
        name = value["parameter"]
        load_files.append([f"{self.data_output}/regression/{name}/X_{data_split}.parquet", f"{self.data_output}/regression/{name}/y_{data_split}.parquet", f"{self.data_output}/regression/{name}/wt_{data_split}.parquet"])
        edit_files.append(f"regression/{name}/wt_{data_split}.parquet")

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

    load_files = {}
    edit_files = {}
    for data_split in ["train","test"]:
      load_files[data_split] = []
      edit_files[data_split] = []

      # density model
      load_files[data_split].append([f"{self.data_output}/density/X_{data_split}.parquet", f"{self.data_output}/density/Y_{data_split}.parquet", f"{self.data_output}/density/wt_{data_split}.parquet"])
      edit_files[data_split].append(f"density/wt_{data_split}.parquet")
      # regression models
      for value in cfg["models"][file_name]["regression_models"]:
        name = value["parameter"]
        load_files[data_split].append([f"{self.data_output}/regression/{name}/X_{data_split}.parquet", f"{self.data_output}/regression/{name}/y_{data_split}.parquet", f"{self.data_output}/regression/{name}/wt_{data_split}.parquet"])
        edit_files[data_split].append(f"regression/{name}/wt_{data_split}.parquet")

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

    # Load config
    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg) 

    # Make yields
    if self.verbose:
      print("- Calculating yields")
    yields = self._GetYields(self.file_name, cfg)

    # Get binned fit input
    #if self.binned_fit_input is not None:
    #  if len(cfg["pois"]) > 1:
    #    raise ValueError("Binned fits do not work for more the one POIs.")
    #  if self.verbose:
    #    print("- Getting inputs for binned fit")
    #  binned_fit_inputs = self._GetBinnedFitInputs(self.file_name, cfg)
    #else:
    #  binned_fit_inputs = None

    # Train/Val split the dataset
    if self.verbose:
      print("- Doing train/test/val splitting")
    self._DoTrainTestValSplit(self.file_name, cfg)

    # Do variations in each of the train/test/test_inf/val datasets
    if self.verbose:
      print("- Calculating variations for each dataset")
    self._DoModelVariations(self.file_name, cfg)

    # Normalise validations datasets to 1
    if self.verbose:
      print("- Normalising validation datasets to 1")
    self._DoValidationNormalisation(self.file_name, cfg, yields)

    # Get effective events of validation datasets
    if self.verbose:
      print("- Getting effective events in each validation dataset")
    eff_events = self._GetValidationEffEvents(self.file_name, cfg)

    # Flatten train/test with from yields
    if self.verbose:
      print("- Flattening train and test dataset")
    self._DoFlattenByYields(self.file_name, cfg, yields)

    if self.verbose:
      print("- Normalising train and test in given selections")
    self._DoNormaliseIn(self.file_name, cfg)

    # Standardise
    if self.verbose:
      print("- Standardising train and test datasets")
    standardisation_parameters = self._GetStandardisationParameters(self.file_name, cfg)
    self._DoStandardisation(self.file_name, cfg, standardisation_parameters)

    # Shuffle
    if self.verbose:
      print("- Shuffling train and test datasets")
    self._DoShuffle(self.file_name, cfg)

    # Write parameters
    if self.verbose:
      print("- Writing parameters")
    self._DoParametersFile(self.file_name, cfg, yields=yields, eff_events=eff_events, standardisation=standardisation_parameters)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add parameters file
    outputs += [f"{self.data_output}/parameters.yaml"]

    # Check if we output Extra
    density_loop = ["X","Y","wt"]
    regression_loop = ["X","y","wt"]
    if "save_extra_columns" in cfg["preprocess"]:
      if self.file_name in cfg["preprocess"]["save_extra_columns"]:
        density_loop += ["Extra"]
        regression_loop += ["Extra"]

    # Add train/test files
    for data_split in ["train","test"]:

      # Add density files
      for k in density_loop:
        outputs += [f"{self.data_output}/density/{k}_{data_split}.parquet"]

      # Add regression files
      for k in regression_loop:
        for value in cfg["models"][self.file_name]["regression_models"]:
          outputs += [f"{self.data_output}/regression/{value['parameter']}/{k}_{data_split}.parquet"]

    # Add validation files
    for data_split in ["val","train_inf","test_inf","full"]:
      for k in density_loop:
        for ind in range(len(GetValidationLoop(cfg, self.file_name, include_rate=True, include_lnN=True))):
          outputs += [f"{self.data_output}/val_ind_{ind}/{k}_{data_split}.parquet"]   

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
      inputs += [f"{self.data_input}/{cfg['models'][self.file_name]['yields']['file']}.parquet"]

    # Add validation base files
    inputs += [f"{self.data_input}/{cfg['validation']['files'][self.file_name]}.parquet"]

    # Add variation base files
    files_in_model = GetFilesInModel(self.file_name, cfg)
    for uf in files_in_model:
      file_name = f"{self.data_input}/{uf}.parquet"
      if file_name not in inputs:
        inputs += [file_name]

    return inputs
