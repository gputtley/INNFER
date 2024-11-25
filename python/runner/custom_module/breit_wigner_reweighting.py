import copy
import os
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from scipy.interpolate import UnivariateSpline

from data_processor import DataProcessor

class breit_wigner_reweighting():

  def __init__(self):
    self.cfg = None
    self.n_samples_per_point = 10
    self.write = True
    self.seed = 21

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

  def _DoWriteDatasets(self, df, X_columns, Y_columns, file_loc="data/", data_split="val", extra_name=""):

    loop_over = {"X":X_columns, "Y":Y_columns, "wt":["wt"]}
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

  def _ChangeBatch(self, df, unique=[], fractions={}):
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

  def _FlatteningSpline(self, df, spline=None):
    df.loc[:,"wt"] /= spline(df.loc[:,"mass"])
    return df

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

    # File location
    file_loc = f"data/{cfg['name']}/ttbar/PreProcess"

    # Open the parameters
    with open(f"{file_loc}/parameters.yaml", 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    ### Get fractions ###

    # full dataprocessor
    dp = DataProcessor(
      [[f"{file_loc}/X_full.parquet", f"{file_loc}/Y_full.parquet", f"{file_loc}/wt_full.parquet",f"{file_loc}/Extra_full.parquet"]],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : parameters,
      }
    )

    # get unique Y and loop through
    unique = dp.GetFull(method="unique")["mass"]

    # get sum of weights and sum of weights squared after BW
    nominal_sum_wt = {}
    sum_wt = {}
    sum_wt_squared = {}
    for transformed_to in unique:
      sum_wt[transformed_to] = {}
      sum_wt_squared[transformed_to] = {}
      for transformed_from in unique:
        sum_wt[transformed_to][transformed_from] = dp.GetFull(method="sum", extra_sel=f"mass=={transformed_from}", functions_to_apply=[partial(self._ApplyBWReweight,m=transformed_to,l=self._TopQuarkWidth(transformed_to))])
        sum_wt_squared[transformed_to][transformed_from] = dp.GetFull(method="sum_w2", extra_sel=f"mass=={transformed_from}", functions_to_apply=[partial(self._ApplyBWReweight,m=transformed_to,l=self._TopQuarkWidth(transformed_to))])

    # derive fractions
    fractions = {}
    for transformed_to in unique:
      fractions[transformed_to] = {}
      for transformed_from in unique:
        fractions[transformed_to][transformed_from] = (sum_wt[transformed_to][transformed_from] / sum_wt_squared[transformed_to][transformed_from]) / (sum_wt[transformed_to][transformed_to] / sum_wt_squared[transformed_to][transformed_to])

    # derive normalisation
    normalised_fractions = {}
    for transformed_to in unique:
      total_sum = np.sum(np.array(list(sum_wt[transformed_to].values())) * np.array(list(fractions[transformed_to].values())))
      normalised_fractions[transformed_to] = {}
      for transformed_from in unique:
        normalised_fractions[transformed_to][transformed_from] = sum_wt[transformed_to][transformed_to] * fractions[transformed_to][transformed_from] / total_sum

    # fit splines for continuous fractioning
    splines = {}
    masses = list(normalised_fractions.keys())
    for transformed_from in unique:
      fractions_to = [normalised_fractions[transformed_to][transformed_from] for transformed_to in unique]
      splines[transformed_from] = UnivariateSpline(masses, fractions_to, s=0, k=1)
      x = np.linspace(min(unique),max(unique),num=10)
      y = splines[transformed_from](x)


    ### Do Discrete Reweighting ###

    for data_split in ["test_inf","val"]:

      # partial dataprocessor
      dp = DataProcessor(
        [[f"{file_loc}/X_{data_split}.parquet", f"{file_loc}/Y_{data_split}.parquet", f"{file_loc}/wt_{data_split}.parquet",f"{file_loc}/Extra_{data_split}.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : parameters,
          "functions" : ["untransform"]
        }
      )

      # Get unique
      unique = dp.GetFull(method="unique")["mass"]

      # Find normalisation
      norm_hist, norm_bins = dp.GetFull(method="histogram", column="mass", bins=100, functions_to_apply = [partial(self._ChangeBatch, unique=unique, fractions=normalised_fractions)])
      normalisation = dict(zip(norm_bins[:-1], 1/norm_hist))

      # Apply reweighting on datasets
      for data_type in ["X","Y","wt"]:
        file_name = f"{file_loc}/{data_type}_{data_split}_bw.parquet"
        if os.path.isfile(file_name):
          os.system(f"rm {file_name}")
      dp.GetFull(
        method=None, 
        functions_to_apply = [
          partial(self._ChangeBatch, unique=unique, fractions=normalised_fractions),
          partial(self._ApplyFractions, fractions=normalisation),
          "transform",
          partial(self._DoWriteDatasets, X_columns=parameters["X_columns"], Y_columns=parameters["Y_columns"], file_loc=file_loc, data_split=data_split, extra_name="_bw")
        ]
      )

      # Recalculate yields
      yields_dp = DataProcessor(
        [[f"{file_loc}/X_{data_split}_bw.parquet", f"{file_loc}/Y_{data_split}_bw.parquet", f"{file_loc}/wt_{data_split}_bw.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : parameters,
          "functions" : ["untransform"]
        }
      )
      
      # Load in yields file
      yield_dataframe = pd.read_parquet(parameters['yield_loc'])
      for ind, mass in enumerate(yield_dataframe.loc[:,"mass"].to_numpy()):
        if mass not in unique: continue
        yield_dataframe.loc[ind, f"sum_wt_squared_{data_split}"] = yields_dp.GetFull(method="sum_w2",extra_sel=f"mass=={mass}")
        yield_dataframe.loc[ind, f"effective_events_{data_split}"] = yields_dp.GetFull(method="n_eff",extra_sel=f"mass=={mass}")

      # Write the information
      if self.write:
        table = pa.Table.from_pandas(yield_dataframe, preserve_index=False)
        pq.write_table(table, parameters["yield_loc"], compression='snappy')
        for data_type in ["X","Y","wt"]:
          file_name = f"{file_loc}/{data_type}_{data_split}_bw.parquet"
          mv_file_name = f"{file_loc}/{data_type}_{data_split}.parquet"
          os.system(f"mv {file_name} {mv_file_name}")

    ### Do Continuous Reweighting ###

    for data_split in ["train","test"]:

      dp = DataProcessor(
        [[f"{file_loc}/X_{data_split}.parquet", f"{file_loc}/Y_{data_split}.parquet", f"{file_loc}/wt_{data_split}.parquet",f"{file_loc}/Extra_{data_split}.parquet"]],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : parameters,
          "functions" : ["untransform"]
        }
      )

      # get unique
      unique = dp.GetFull(method="unique")["mass"]

      # Apply reweighting on datasets
      for data_type in ["X","Y","wt"]:
        file_name = f"{file_loc}/{data_type}_{data_split}_bw.parquet"
        if os.path.isfile(file_name):
          os.system(f"rm {file_name}")

      # Make flattening
      np.random.seed(self.seed)
      hist, bins =  dp.GetFull(
        method="histogram", 
        column="mass",
        bins=100,
        functions_to_apply = [
          partial(self._ChangeBatchContinuous, min_sample=min(unique), max_sample=max(unique), fraction_splines=splines),
        ]
      )
      flattening_spline = UnivariateSpline([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)], hist, s=0, k=1)

      np.random.seed(self.seed)
      dp.GetFull(
        method=None, 
        functions_to_apply = [
          partial(self._ChangeBatchContinuous, min_sample=min(unique), max_sample=max(unique), fraction_splines=splines),
          partial(self._FlatteningSpline, spline=flattening_spline),
          "transform",
          partial(self._DoWriteDatasets, X_columns=parameters["X_columns"], Y_columns=parameters["Y_columns"], file_loc=file_loc, data_split=data_split, extra_name="_bw")
        ]
      )

      if self.write:
        for data_type in ["X","Y","wt"]:
          file_name = f"{file_loc}/{data_type}_{data_split}_bw.parquet"
          mv_file_name = f"{file_loc}/{data_type}_{data_split}.parquet"
          os.system(f"mv {file_name} {mv_file_name}")

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
    inputs = []
    return inputs