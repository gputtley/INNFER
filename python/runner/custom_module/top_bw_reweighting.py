import copy
import os
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns

from functools import partial
from scipy.interpolate import UnivariateSpline

from data_processor import DataProcessor
from plotting import plot_histograms
from useful_functions import MakeDirectories

class top_bw_reweighting():

  def __init__(self):
    self.cfg = None
    self.options = {}
    self.batch_size = 10**7
    self.number_of_shuffles = 10

  def _BW(self, s, l=1.32, m=172.5):
    k = 1
    return k/((s-(m**2))**2 + (m*l)**2)

  def _ApplyBWReweight(self, df, m=172.5, l=1.32):
    df.loc[:,"wt"] *= self._BW(df.loc[:,"mass_had_top"]**2,l=l, m=m)/self._BW(df.loc[:,"mass_had_top"]**2,l=self._TopQuarkWidth(df.loc[:,"mass"]), m=df.loc[:,"mass"]) 
    return df

  def _TopQuarkWidth(self, m):
    return (0.0270*m) - 3.3455

  def _ApplyFractions(self, df, fractions={}):
    df.loc[:,"wt"] *= df.loc[:,"mass"].map(fractions)
    return df  

  def _DoWriteDatasets(self, df, X_columns, Y_columns, file_loc="data/", data_split="val", extra_columns=[], extra_name=""):

    loop_over = {"X":X_columns, "Y":Y_columns, "wt":["wt"]}
    if len(extra_columns) > 0:
      loop_over["Extra"] = extra_columns
    for data_type, columns in loop_over.items():
      if len(df) == 0: continue
      table = pa.Table.from_pandas(df.loc[:, sorted(columns)], preserve_index=False)
      file_path = f"{file_loc}/{data_type}_{data_split}{extra_name}.parquet"
      if os.path.isfile(file_path):
        combined_table = pa.concat_tables([pq.read_table(file_path), table])
        pq.write_table(combined_table, file_path, compression='snappy')
      else:
        pq.write_table(table, file_path, compression='snappy')
    return df

  def _ChangeBatchDiscrete(self, df, unique=[], fractions={}):
    copy_df = copy.deepcopy(df)
    first = True
    for mass in unique:
      tmp = self._ApplyBWReweight(copy.deepcopy(copy_df), m=mass, l=self._TopQuarkWidth(mass))
      tmp = self._ApplyFractions(tmp, fractions=fractions[mass])
      tmp.loc[:,"mass"] = mass
      if first:
        df = copy.deepcopy(tmp)
        first = False
      else:
        df = pd.concat([df, tmp], ignore_index=True)
    return df

  def _ChangeBatchContinuous(self, df, min_sample=0.0, max_sample=1.0, fraction_splines={}):
    copy_df = copy.deepcopy(df)
    first = True
    sampled = np.random.uniform(low=min_sample, high=max_sample, size=(len(copy_df), self.n_samples_per_point))
    for i, mass in enumerate(sampled.T):
      tmp = self._ApplyBWReweight(copy_df, m=mass, l=self._TopQuarkWidth(mass))
      for k, v in fraction_splines.items():
        indices = (tmp.loc[:,"mass"]==k)
        tmp.loc[indices,"wt"] *= v(mass[indices])

      tmp.loc[:,"mass"] = mass
      if first:
        df = copy.deepcopy(tmp)
        first = False
      else:
        df = pd.concat([df, tmp], ignore_index=True)
    return df

  def _DoShuffle(self, data_splits=["train","test"], data_output="data/"):

    for data_split in data_splits:
      for i in ["X","Y","wt","Extra"]:
        name = f"{data_output}/{i}_{data_split}.parquet"
        if not os.path.isfile(name): continue
        shuffle_name = f"{data_output}/{i}_{data_split}_shuffled.parquet"
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

  def _FlatteningSpline(self, df, spline=None):
    df.loc[:,"wt"] /= spline(df.loc[:,"mass"])
    return df

  def CalculateOptimalFractions(self):

    print("- Calculating optimal fractions")

    # full dataprocessor
    dp = DataProcessor(
      [[f"{self.file_loc}/X_full.parquet", f"{self.file_loc}/Y_full.parquet", f"{self.file_loc}/wt_full.parquet",f"{self.file_loc}/Extra_full.parquet"]],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : self.parameters,
      },
      batch_size=self.batch_size,
    )

    # get unique Y and loop through
    unique = dp.GetFull(method="unique")["mass"]

    # get sum of weights and sum of weights squared after BW
    nominal_sum_wt = {}
    sum_wt = {}
    sum_wt_squared = {}
    for transformed_to in self.discrete_masses:
      sum_wt[transformed_to] = {}
      sum_wt_squared[transformed_to] = {}
      for transformed_from in unique:
        sum_wt[transformed_to][transformed_from] = dp.GetFull(method="sum", extra_sel=f"mass=={transformed_from}", functions_to_apply=[partial(self._ApplyBWReweight,m=transformed_to,l=self._TopQuarkWidth(transformed_to))])
        sum_wt_squared[transformed_to][transformed_from] = dp.GetFull(method="sum_w2", extra_sel=f"mass=={transformed_from}", functions_to_apply=[partial(self._ApplyBWReweight,m=transformed_to,l=self._TopQuarkWidth(transformed_to))])

    # derive fractions
    fractions = {}
    for transformed_to in self.discrete_masses:
      fractions[transformed_to] = {}
      for transformed_from in unique:
        fractions[transformed_to][transformed_from] = (sum_wt[transformed_to][transformed_from] / sum_wt_squared[transformed_to][transformed_from])

    # derive normalisation
    normalised_fractions = {}
    for transformed_to in self.discrete_masses:
      normalised_fractions[transformed_to] = {}
      total_sum = np.sum(np.array(list(sum_wt[transformed_to].values())) * np.array(list(fractions[transformed_to].values())))
      for transformed_from in unique:
        normalised_fractions[transformed_to][transformed_from] = fractions[transformed_to][transformed_from] / total_sum

    # fit splines for continuous fractioning
    splines = {}
    masses = list(normalised_fractions.keys())
    for transformed_from in unique:
      fractions_to = [normalised_fractions[transformed_to][transformed_from] for transformed_to in self.discrete_masses]
      splines[transformed_from] = UnivariateSpline(masses, fractions_to, s=0, k=1)

    return normalised_fractions, splines

  def DoDiscreteReweighting(self, normalised_fractions):

    for data_split in self.discrete_reweighting:

      print(f"- Doing discrete reweighting for {data_split}")

      if data_split in ["train","test"]:
        discrete_masses = self.discrete_masses_train
      elif data_split == "full":
        discrete_masses = self.discrete_masses
      else:
        discrete_masses = self.discrete_masses_inference


      if data_split == "full":
        initial_functions = []
        final_functions = []
      else:
        initial_functions = ["untransform"]
        final_functions = ["transform"]

      # partial dataprocessor
      dp = DataProcessor(
        [[f"{self.file_loc}/X_{data_split}.parquet", f"{self.file_loc}/Y_{data_split}.parquet", f"{self.file_loc}/wt_{data_split}.parquet",f"{self.file_loc}/Extra_{data_split}.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : self.parameters,
          "functions" : initial_functions
        },
        batch_size=self.batch_size,
      )

      # Get unique
      unique = dp.GetFull(method="unique")["mass"]

      # Find normalisation
      norm_hist, norm_bins = dp.GetFull(method="histogram", column="mass", bins=100, functions_to_apply = [partial(self._ChangeBatchDiscrete, unique=discrete_masses, fractions=normalised_fractions)])
      norm_hist_before, norm_bins_before = dp.GetFull(method="histogram", column="mass", bins=100)
      non_zero_indices = np.where(norm_hist_before != 0)
      norm_hist_before = norm_hist_before[non_zero_indices]
      norm_bins_before = norm_bins_before[non_zero_indices]
      norm_before_spline = UnivariateSpline(norm_bins_before, norm_hist_before, s=0, k=1)
      norm_hist_before = [norm_before_spline(norm_bin) for norm_bin in norm_bins[:-1]]
      normalisation = dict(zip(norm_bins[:-1], norm_hist_before/norm_hist))

      # Apply reweighting on datasets
      for data_type in ["X","Y","wt","Extra"]:
        file_name = f"{self.file_loc}/{data_type}_{data_split}_bw.parquet"
        if os.path.isfile(file_name):
          os.system(f"rm {file_name}")

      functions_to_apply = [          
        partial(self._ChangeBatchDiscrete, unique=discrete_masses, fractions=normalised_fractions),
        partial(self._ApplyFractions, fractions=normalisation)
      ]
      functions_to_apply += final_functions
      functions_to_apply += [partial(self._DoWriteDatasets, X_columns=self.parameters["X_columns"], Y_columns=self.parameters["Y_columns"], extra_columns=self.parameters["Extra_columns"], file_loc=self.file_loc, data_split=data_split, extra_name="_bw")]

      dp.GetFull(
        method=None, 
        functions_to_apply = functions_to_apply
      )

      # Recalculate yields
      yields_dp = DataProcessor(
        [[f"{self.file_loc}/X_{data_split}_bw.parquet", f"{self.file_loc}/Y_{data_split}_bw.parquet", f"{self.file_loc}/wt_{data_split}_bw.parquet", f"{self.file_loc}/Extra_{data_split}_bw.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : self.parameters,
          "functions" : initial_functions
        },
        batch_size=self.batch_size,
      )
      
      print(f"- Recalculating yields for {data_split}")

      # Load in yields file
      yield_dataframe = pd.read_parquet(self.parameters['yield_loc'])
      unique_combinations = yield_dataframe.loc[:,self.parameters["Y_columns"]]
      unique_sum_w = yields_dp.GetFull(method="sum_w_unique_columns", unique_combinations=unique_combinations)
      unique_sum_w2 = yields_dp.GetFull(method="sum_w2_unique_columns", unique_combinations=unique_combinations)
      unique_count = yields_dp.GetFull(method="count_unique_columns", unique_combinations=unique_combinations)
      if data_split != "full":
        yield_dataframe.loc[:, f"effective_events_{data_split}"] = np.where(unique_sum_w2.loc[:,"sum_w2"] == 0, 0, unique_sum_w.loc[:,"sum_w"]**2 / unique_sum_w2.loc[:,"sum_w2"])
        yield_dataframe.loc[:, f"sum_wt_squared_{data_split}"] = np.where(yield_dataframe.loc[:, f"effective_events_{data_split}"] == 0, 0, yield_dataframe.loc[:, f"yields_{data_split}"]**2 / yield_dataframe.loc[:, f"effective_events_{data_split}"])
        yield_dataframe.loc[:, f"length_{data_split}"] = unique_count.loc[:,"count"]
      else:
        yield_dataframe.loc[:, f"sum_wt_squared"] = unique_sum_w2.loc[:,"sum_w2"]
        yield_dataframe.loc[:, f"effective_events"] =  np.where(unique_sum_w2.loc[:,"sum_w2"] == 0, 0, unique_sum_w.loc[:,"sum_w"]**2 / unique_sum_w2.loc[:,"sum_w2"])
        yield_dataframe.loc[:, f"length"] = unique_count.loc[:,"count"]

      # Write the information
      if self.write:
        table = pa.Table.from_pandas(yield_dataframe, preserve_index=False)
        pq.write_table(table, self.parameters["yield_loc"], compression='snappy')
        for data_type in ["X","Y","wt","Extra"]:
          file_name = f"{self.file_loc}/{data_type}_{data_split}_bw.parquet"
          mv_file_name = f"{self.file_loc}/{data_type}_{data_split}.parquet"
          os.system(f"mv {file_name} {mv_file_name}")

    # Rewrite parameters file with new unique value
    if self.write:
      with open(f"{self.file_loc}/parameters.yaml", 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
      parameters["unique_Y_values"]["mass"] = self.discrete_masses_inference
      with open(f"{self.file_loc}/parameters.yaml", 'w') as yaml_file:
        yaml.dump(parameters, yaml_file)

  def DoContinuousReweighting(self, splines):

    for data_split in self.continuous_reweighting:

      print(f"- Doing continuous reweighting for {data_split}")

      dp = DataProcessor(
        [[f"{self.file_loc}/X_{data_split}.parquet", f"{self.file_loc}/Y_{data_split}.parquet", f"{self.file_loc}/wt_{data_split}.parquet",f"{self.file_loc}/Extra_{data_split}.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : self.parameters,
          "functions" : ["untransform"]
        },
        batch_size=self.batch_size,
      )

      # Apply reweighting on datasets
      for data_type in ["X","Y","wt","Extra"]:
        file_name = f"{self.file_loc}/{data_type}_{data_split}_bw.parquet"
        if os.path.isfile(file_name):
          os.system(f"rm {file_name}")

      # Make flattening
      np.random.seed(self.seed)
      hist, bins =  dp.GetFull(
        method="histogram", 
        column="mass",
        bins=100,
        functions_to_apply = [
          partial(self._ChangeBatchContinuous, min_sample=self.min_mass, max_sample=self.max_mass, fraction_splines=splines),
        ]
      )
      flattening_spline = UnivariateSpline([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)], hist, s=0, k=1)

      np.random.seed(self.seed)
      dp.GetFull(
        method=None, 
        functions_to_apply = [
          partial(self._ChangeBatchContinuous, min_sample=self.min_mass, max_sample=self.max_mass, fraction_splines=splines),
          partial(self._FlatteningSpline, spline=flattening_spline),
          "transform",
          partial(self._DoWriteDatasets, X_columns=self.parameters["X_columns"], Y_columns=self.parameters["Y_columns"], extra_columns=self.parameters["Extra_columns"], file_loc=self.file_loc, data_split=data_split, extra_name="_bw")
        ]
      )

      if self.write:
        for data_type in ["X","Y","wt","Extra"]:
          file_name = f"{self.file_loc}/{data_type}_{data_split}_bw.parquet"
          mv_file_name = f"{self.file_loc}/{data_type}_{data_split}.parquet"
          os.system(f"mv {file_name} {mv_file_name}")

  def PlotReweighting(self, normalised_fractions):

    print("- Plotting optimally reweighted samples")

    dp = DataProcessor(
      [[f"{self.file_loc}/X_full.parquet", f"{self.file_loc}/Y_full.parquet", f"{self.file_loc}/wt_full.parquet",f"{self.file_loc}/Extra_full.parquet"]],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : self.parameters,
      },
      batch_size=self.batch_size,
    )

    unique = dp.GetFull(method="unique")
    for col in unique.keys():
      if col == "mass": continue
      
      hists = []
      hist_names = []
      hist_uncerts = []
      drawstyles = []
      colours = []
      error_bar_hists = []
      error_bar_hist_uncerts = []
      error_bar_hist_names = []

      bins = dp.GetFull(
        method="bins_with_equal_spacing", 
        bins=40,
        column=col,
        ignore_quantile=0.02
      )

      colour_list = sns.color_palette("Set2", len(unique["mass"]))


      for i, mass in enumerate(unique["mass"]):

        var_hist, var_hist_uncert, _ = dp.GetFull(
          method="histogram_and_uncert", 
          extra_sel=f"mass=={mass}",
          bins=bins,
          column=col,
        )

        integral = np.sum(var_hist*(bins[1]-bins[0]))
        var_hist /= integral
        var_hist_uncert /= integral
        hists.append(var_hist)
        hist_uncerts.append(var_hist_uncert)
        hist_names.append(r"$m_{t}$ = " + f"{mass} GeV")
        drawstyles.append("steps-mid")
        colours.append(colour_list[i])

        bw_reweighted_hist, bw_reweighted_hist_uncert, _ = dp.GetFull(
          method="histogram_and_uncert", 
          bins=bins,
          column=col,
          functions_to_apply=[partial(self._ApplyBWReweight,m=mass,l=self._TopQuarkWidth(mass)),partial(self._ApplyFractions, fractions=normalised_fractions[mass])]
        )

        integral = np.sum(bw_reweighted_hist*(bins[1]-bins[0]))
        bw_reweighted_hist /= integral
        bw_reweighted_hist_uncert /= integral
        error_bar_hists.append(bw_reweighted_hist)
        error_bar_hist_uncerts.append(bw_reweighted_hist_uncert)
        error_bar_hist_names.append(None)

      with open(self.cfg, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
      plot_dir = f"plots/{cfg['name']}/ttbar/top_bw_reweighting"
      MakeDirectories(plot_dir)

      plot_histograms(
        np.array(bins[:-1]), 
        hists, 
        hist_names, 
        hist_errs = hist_uncerts,
        error_bar_hists = error_bar_hists,
        error_bar_hist_errs = error_bar_hist_uncerts,
        error_bar_names = error_bar_hist_names,
        drawstyle=drawstyles, 
        colors=colours, 
        name=f"{plot_dir}/bw_reweighted_{col}", 
        x_label=col, 
        y_label="Density"
      )


  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    self.write = True if "write" not in self.options else self.options["write"].strip() == "True"
    self.seed = 21 if "seed" not in self.options else int(self.options["seed"])

    self.discrete_masses_inference = [171.5,172.0,172.5,173.0,173.5] if "discrete_masses" not in self.options else [float(i) for i in (self.options["discrete_masses"].split(",") if "," in self.options["discrete_masses"] else [])]
    #self.discrete_masses_inference = [171.5,172.5,173.5] if "discrete_masses" not in self.options else [float(i) for i in (self.options["discrete_masses"].split(",") if "," in self.options["discrete_masses"] else [])]

    self.discrete_mode = False if "discrete_mode" not in self.options else self.options["discrete_mode"].strip() == "True"

    self.continuous_mode = False if "continuous_mode" not in self.options else self.options["continuous_mode"].strip() == "True"

    if self.discrete_mode:

      self.continuous_reweighting = []
      self.discrete_reweighting = ["train","test","test_inf","val","full"]
      self.discrete_masses_train = [166.5,169.5,171.5,172.5,173.5,175.5,178.5]
      self.make_copy = {}

    elif self.continuous_mode:

      self.discrete_reweighting = ["test_inf","val","full"]
      self.discrete_masses_train = [166.5,169.5,171.5,172.5,173.5,175.5,178.5]
      self.continuous_reweighting = ["train","test"]
      self.n_samples_per_point = 7
      self.min_mass = 166.5
      self.max_mass = 178.5
      self.make_copy = {"train":"train_inf"}

    else:

      self.continuous_reweighting = ["train","test"] if "continuous_reweighting" not in self.options else self.options["continuous_reweighting"].split(",") if "," in self.options["continuous_reweighting"] else []
      self.n_samples_per_point = 20 if "n_samples_per_point" not in self.options else int(self.options["n_samples_per_point"])
      self.min_mass = 166.5 if "min_mass" not in self.options else float(self.options["min_mass"])
      self.max_mass = 178.5 if "max_mass" not in self.options else float(self.options["max_mass"])

      self.discrete_reweighting = ["train_inf","test_inf","val","full"] if "discrete_reweighting" not in self.options else self.options["discrete_reweighting"].split(",") if "," in self.options["discrete_reweighting"] else []
      self.discrete_masses_train = [166.5,169.5,171.5,172.5,173.5,175.5,178.5] if "discrete_masses_train_test" not in self.options else [float(i) for i in (self.options["discrete_masses_train_test"].split(",") if "," in self.options["discrete_masses_train_test"] else [])]
      self.make_copy = {"train":"train_inf"} if "make_copy" not in self.options else {i.split(":")[0]:i.split(":")[1] for i in self.options["make_copy"].split(",")}

    self.discrete_masses = sorted(list(set(self.discrete_masses_inference + self.discrete_masses_train)))


  def Run(self):
    """
    Run the code utilising the worker classes
    """    
    # Open the config
    with open(self.cfg, 'r') as yaml_file:
      cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # File location
    self.file_loc = f"data/{cfg['name']}/ttbar/PreProcess"

    # Open the parameters
    with open(f"{self.file_loc}/parameters.yaml", 'r') as yaml_file:
      self.parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Get fractions
    normalised_fractions, splines = self.CalculateOptimalFractions()

    # Plot reweighting
    self.PlotReweighting(normalised_fractions)

    # Make copies
    for data_split, copy_to in self.make_copy.items():
      for data_type in ["X","Y","wt","Extra"]:
        file_name = f"{self.file_loc}/{data_type}_{data_split}.parquet"
        if os.path.isfile(file_name):
          os.system(f"cp {file_name} {self.file_loc}/{data_type}_{copy_to}.parquet")

    # Do discrete reweighting
    self.DoDiscreteReweighting(normalised_fractions)

    # Do continuous reweighting
    self.DoContinuousReweighting(splines)

    # Shuffle train and test
    if self.write:
      print("- Shuffling the dataset")
      self._DoShuffle(data_splits=["train","test"], data_output=self.file_loc)

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    return []

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    return []