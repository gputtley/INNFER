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
from plotting import plot_histograms, plot_histograms_with_ratio
from python.worker.useful_functions import GetParametersInModel
from useful_functions import MakeDirectories, LoadConfig, GetValidationLoop, GetCategoryLoop, GetSplitDensityModel, GetModelLoop, SampleFlatTop

data_dir = str(os.getenv("PREP_DATA_DIR"))
plots_dir = str(os.getenv("PLOTS_DIR"))
models_dir = str(os.getenv("MODELS_DIR"))

class top_bw_fractioning():

  def __init__(self):
    """
    A class to preprocess the datasets and produce the data
    parameters yaml file as well as the train, test and
    validation datasets.
    """
    # Required input which is the location of a file
    self.cfg = None
    self.options = {}
    self.batch_size = 10**7


  def _ApplyBWReweight(self, df, m=172.5, l=1.32):
    """
    Apply the BW reweighting to the dataframe.
    Args:
        df (pd.DataFrame): The input dataframe.
        m (float): The mass of the top quark.
        l (float): The width of the top quark. 
    """
    # Apply the BW reweighting

    mask = df.loc[:,self.gen_mass] > 0
    df.loc[mask,"wt"] *= self._BW(df.loc[mask,self.gen_mass]**2,l=l, m=m)/self._BW(df.loc[mask,self.gen_mass]**2,l=self._TopQuarkWidth(df.loc[mask,self.mass_name]), m=df.loc[mask,self.mass_name]) 
    if self.gen_mass_other is not None:
      mask = df.loc[:,self.gen_mass_other] > 0
      df.loc[mask,"wt"] *= self._BW(df.loc[mask,self.gen_mass_other]**2,l=l, m=m)/self._BW(df.loc[mask,self.gen_mass_other]**2,l=self._TopQuarkWidth(df.loc[mask,self.mass_name]), m=df.loc[mask,self.mass_name])
    return df


  def _ApplyFractions(self, df, fractions={}):
    """
    Apply the fractions to the dataframe.
    Args:
        df (pd.DataFrame): The input dataframe.
        fractions (dict): The fractions to apply.
    """
    # Apply the fractions
    df.loc[:,"wt"] *= df.loc[:,self.mass_name].map(fractions)
    return df  


  def _BW(self, s, l=1.32, m=172.5):
    """
    Calculate the Breit-Wigner distribution.
    Args:
        s (float): The mass squared.
        l (float): The width of the top quark.
        m (float): The mass of the top quark.
    Returns:
        float: The Breit-Wigner distribution.
    """
    # Calculate the Breit-Wigner distribution
    k = 1
    return k/((s-(m**2))**2 + (m*l)**2)


  def _CalculateOptimalFractions(self, base_file, wt_func, selection=None):
    """
    Calculate the optimal fractions for the given base file and weight function.
    Args:
        base_file (str): The base file to use.
        wt_func (str): The weight function to use.
    Returns:
        tuple: A tuple containing the normalised fractions and the splines.
    """

    print("- Calculating optimal fractions")

    # full dataprocessor
    dp = DataProcessor(
      [[base_file]],
      "parquet",
      batch_size=self.batch_size,
      options = {
        "wt_name" : "wt",
        "selection" : selection
      }
    )

    # calculate weight
    def apply_wt(df, wt_func):
      if self.bw_mass_name not in df.columns:
        df.loc[:,self.bw_mass_name] = 172.5*np.ones(len(df))
      df.loc[:,"wt"] = df.eval(wt_func)
      return df
    apply_wt_partial = partial(apply_wt, wt_func=wt_func)

    # get unique Y and loop through
    unique = dp.GetFull(method="unique")[self.mass_name]

    # get nominal sum of weights
    nominal_sum_wt = []
    for transformed_from in unique:
      nominal_sum_wt.append(dp.GetFull(
          method = "sum", 
          extra_sel = f"{self.mass_name}=={transformed_from}", 
          functions_to_apply = [
            apply_wt_partial,
          ]
        )
      )

    # get nominal sum of weights spline
    nominal_sum_wt_splines = UnivariateSpline(unique, nominal_sum_wt, s=0, k=1)

    # get sum of weights and sum of weights squared after BW
    sum_wt = {}
    sum_wt_squared = {}
    for transformed_to in self.transformed_to_masses:
      sum_wt[transformed_to] = {}
      sum_wt_squared[transformed_to] = {}
      for transformed_from in unique:

        sum_wt[transformed_to][transformed_from] = dp.GetFull(
          method = "sum", 
          extra_sel = f"{self.mass_name}=={transformed_from}", 
          functions_to_apply = [
            apply_wt_partial,
            partial(self._ApplyBWReweight,m=transformed_to,l=self._TopQuarkWidth(transformed_to))
          ]
        )
        sum_wt_squared[transformed_to][transformed_from] = dp.GetFull(
          method = "sum_w2", 
          extra_sel = f"{self.mass_name}=={transformed_from}", 
          functions_to_apply = [
            apply_wt_partial,
            partial(self._ApplyBWReweight,m=transformed_to,l=self._TopQuarkWidth(transformed_to))
          ]
        )

    # Effective events
    eff_events = {}
    for transformed_to in self.transformed_to_masses:
      eff_events[transformed_to] = {}
      for transformed_from in unique:
        if sum_wt_squared[transformed_to][transformed_from] > 0:
          eff_events[transformed_to][transformed_from] = (sum_wt[transformed_to][transformed_from]**2) / sum_wt_squared[transformed_to][transformed_from]
        else:
          eff_events[transformed_to][transformed_from] = 0.0

    # Total effective events for transformed_to
    total_eff_events = {}
    for transformed_to in self.transformed_to_masses:
      total_eff_events[transformed_to] = np.sum(np.array(list(eff_events[transformed_to].values())))

    # Fraction of effective events
    eff_events_fraction = {}
    for transformed_to in self.transformed_to_masses:
      eff_events_fraction[transformed_to] = {}
      for transformed_from in unique:
        if total_eff_events[transformed_to] > 0:
          eff_events_fraction[transformed_to][transformed_from] = eff_events[transformed_to][transformed_from] / total_eff_events[transformed_to]
        else:
          eff_events_fraction[transformed_to][transformed_from] = 0.0

    # derive fractions
    fractions = {}
    for transformed_to in self.transformed_to_masses:
      fractions[transformed_to] = {}
      for transformed_from in unique:
        if eff_events_fraction[transformed_to][transformed_from] > self.ignore_quantile:
          fractions[transformed_to][transformed_from] = (sum_wt[transformed_to][transformed_from] / sum_wt_squared[transformed_to][transformed_from])
        else:
          fractions[transformed_to][transformed_from] = 0.0

    # derive normalisation
    normalised_fractions = {}
    for transformed_to in self.transformed_to_masses:
      normalised_fractions[transformed_to] = {}
      total_sum = np.sum(np.array(list(sum_wt[transformed_to].values())) * np.array(list(fractions[transformed_to].values())))
      for transformed_from in unique:
        normalised_fractions[transformed_to][transformed_from] = fractions[transformed_to][transformed_from] * nominal_sum_wt_splines(transformed_to) / total_sum


    # fit splines for continuous fractioning
    splines = {}
    masses = list(normalised_fractions.keys())
    for transformed_from in unique:
      fractions_to = [normalised_fractions[transformed_to][transformed_from] for transformed_to in self.transformed_to_masses]
      
      splines[transformed_from] = UnivariateSpline(masses, fractions_to, s=0, k=1)
      # Cap spline function to a minimum of 0
      def SplineWithMinZero(x, spline_func=splines[transformed_from]):
        result = spline_func(x)
        result[result < 0] = 0.0
        return result
      splines[transformed_from] = partial(SplineWithMinZero, spline_func=splines[transformed_from])

    return normalised_fractions, splines


  def _DoReweighting(self, splines, files, parameters, file_type, shift_file, shift_column):
    """
    Apply the reweighting to the given files.
    Args:
        splines (dict): The splines to use for the reweighting.
        files (list): The list of files to reweight.
        parameters (dict): The parameters to use for the reweighting.
        file_type (str): The type of file to reweight.
        shift_file (str): The file to shift.
        shift_column (str): The column to shift.
    """

    print(f"- Doing fractioning for {files}")

    if file_type == "density":
      parameters_dict = parameters["density"]
      functions = ["untransform"]
      after_function = ["transform"]
    elif file_type == "regression":
      parameters_dict = parameters["regression"][shift_file.split("/")[-2]]
      functions = ["untransform"]
      after_function = ["transform"]
    elif file_type == "classifier":
      parameters_dict = parameters["classifier"][shift_file.split("/")[-2]]
      functions = ["untransform"]
      after_function = ["transform"]
    elif file_type == "validation":
      parameters_dict = {}
      functions = []
      after_function = []


    dp = DataProcessor(
      [files],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : parameters_dict,
        "functions" : functions
      },
      batch_size=self.batch_size,
    )

    #wt_file = copy.deepcopy(shift_file)
    #wt_alt = shift_file.replace(".parquet","_bw_fractioning.parquet")
    #if os.path.isfile(wt_alt):
    #  os.system(f"rm {wt_alt}")

    new_file_names = []
    column_names = []
    for shift_file in files:
      new_file_name = shift_file.replace(".parquet","_bw_fractioning.parquet")
      new_file_names.append(new_file_name)
      column_dp = DataProcessor(
        [shift_file],
        "parquet",
        options = {},
        batch_size=self.batch_size,
      )
      column_names.append(column_dp.LoadNextBatch().columns.tolist())

      
      if os.path.isfile(new_file_name):
        os.system(f"rm {new_file_name}")


    def ApplyFractions(df, fraction_splines={}, shift_column="wt"):
      if self.bw_mass_name not in df.columns:
        df.loc[:,self.bw_mass_name] = 172.5
      for k, v in fraction_splines.items():
        indices = (df.loc[:,self.mass_name]==k)
        df.loc[indices,shift_column] *= v(df.loc[indices,self.bw_mass_name].to_numpy())
      #return df.loc[:,[shift_column]]
      return df.loc[(df.loc[:,shift_column]!=0),:]


    dp.GetFull(
      method=None, 
      functions_to_apply = [partial(ApplyFractions, fraction_splines=splines, shift_column=shift_column)] + after_function+ [partial(self._WriteDataset, file_name=fn, columns=column_names[ind]) for ind, fn in enumerate(new_file_names)],
    )

    if self.write:
      for ind, shift_file in enumerate(files):
        col_file = shift_file
        col_alt = new_file_names[ind]
        if os.path.isfile(col_alt):
          os.system(f"mv {col_alt} {col_file}")


  '''
  def _FlattenTraining(self, file_loc, file_split):
    """
    Flatten the training data by applying a histogram reweighting.
    Args:
        file_loc (str): The location of the files.
        file_split (list): The list of files to flatten.
    """

    dp = DataProcessor(
      [f"{file_loc}/{k}_train.parquet" for k in file_split if os.path.isfile(f"{file_loc}/{k}_train.parquet")],
      "parquet",
      batch_size=self.batch_size,
      options = {
        "wt_name" : "wt"
      }      
    )
    hist, bins = dp.GetFull(method="histogram", column=self.bw_mass_name, bins=40, ignore_quantile=0.0)
    for data_split in ["train","test"]:
      wt_file = f"{file_loc}/wt_{data_split}.parquet"
      wt_flat_file = wt_file.replace(".parquet","_flattened.parquet")
      dp = DataProcessor(
        [f"{file_loc}/{k}_{data_split}.parquet" for k in file_split],
        "parquet",
        batch_size=self.batch_size,
        options = {
          "wt_name" : "wt"
        }      
      ) 
      def apply_flattening(df, hist, bins):
        if self.bw_mass_name not in df.columns:
          df.loc[:,self.bw_mass_name] = 172.5
        for ind in range(len(hist)):
          df.loc[((df.loc[:,self.bw_mass_name]>=bins[ind]) & (df.loc[:,self.bw_mass_name]<bins[ind+1])),"wt"] /= hist[ind]     
        return df.loc[:,["wt"]]
      apply_flattening_partial = partial(apply_flattening, hist=hist, bins=bins)
      if os.path.isfile(wt_flat_file): os.system(f"rm {wt_flat_file}")
      dp.GetFull(
        method=None,
        functions_to_apply = [
          apply_flattening_partial,
          partial(self._WriteDataset, file_name=wt_flat_file)
        ]
      )
      if self.write:
        if os.path.isfile(wt_flat_file): os.system(f"mv {wt_flat_file} {wt_file}") 
  '''

  def _FlattenTraining(self, file_loc, file_split, parameters, distribution=None):
    """
    Flatten the training data by applying a histogram reweighting.
    Args:
        file_loc (str): The location of the files.
        file_split (list): The list of files to flatten.
    """

    dp = DataProcessor(
      [f"{file_loc}/{k}_train.parquet" for k in file_split if os.path.isfile(f"{file_loc}/{k}_train.parquet")],
      "parquet",
      batch_size=self.batch_size,
      options = {
        "wt_name" : "wt",
        "functions" : ["untransform"],
        "parameters" : parameters,
      }      
    )
    hist, bins = dp.GetFull(method="histogram", column=self.bw_mass_name, bins=40, ignore_quantile=0.0)

    # Check what distribution we want to flatten to
    if distribution is None:
      hist_to = (np.sum(hist)/len(hist))*np.ones(len(hist))
    elif distribution["type"] == "continuous":
      hist_to = (np.sum(hist)/len(hist))*np.ones(len(hist))
    elif distribution["type"] == "flat_top":
      samples = SampleFlatTop(10**7, (distribution["range"][0], distribution["range"][1]), distribution["other"]["sigma_out"])
      hist_to, _ = np.histogram(samples, bins=bins)
      hist_to = hist_to.astype(float)
      hist_to *= np.sum(hist)/np.sum(hist_to)

    for data_split in ["train","test"]:
      wt_file = f"{file_loc}/wt_{data_split}.parquet"
      wt_flat_file = wt_file.replace(".parquet","_flattened.parquet")
      dp = DataProcessor(
        [f"{file_loc}/{k}_{data_split}.parquet" for k in file_split],
        "parquet",
        batch_size=self.batch_size,
        options = {
          "wt_name" : "wt",
          "functions" : ["untransform"],
          "parameters" : parameters,
        }      
      ) 
      def apply_flattening(df, hist, hist_to, bins):
        if self.bw_mass_name not in df.columns:
          df.loc[:,self.bw_mass_name] = 172.5
        for ind in range(len(hist)):
          if hist[ind] > 0:
            df.loc[((df.loc[:,self.bw_mass_name]>=bins[ind]) & (df.loc[:,self.bw_mass_name]<bins[ind+1])),"wt"] *= hist_to[ind]/hist[ind]     
        return df.loc[:,["wt"]]
      apply_flattening_partial = partial(apply_flattening, hist=hist, hist_to=hist_to, bins=bins)
      if os.path.isfile(wt_flat_file): os.system(f"rm {wt_flat_file}")
      dp.GetFull(
        method=None,
        functions_to_apply = [
          apply_flattening_partial,
          partial(self._WriteDataset, file_name=wt_flat_file)
        ]
      )
      if self.write:
        if os.path.isfile(wt_flat_file): os.system(f"mv {wt_flat_file} {wt_file}") 


  def _GetFiles(self, cfg, category, skip_copy=False):
    """
    Get the files to be used for the reweighting.
    Args:
        cfg (dict): The configuration dictionary.
    Returns:
        tuple: A tuple containing the files, shift files, and shift columns.
    """

    files = {"density":[], "regression":[], "classifier":[], "validation":[]}
    shift_files = {"density":[], "regression":[], "classifier":[], "validation":[]}
    shift_columns = {"density":[], "regression":[], "classifier":[], "validation":[]}

    density_loop = ["X","Y","wt"]
    regression_loop = ["X","y","wt"]
    classifier_loop = ["X","y","wt"]
    if "save_extra_columns" in cfg["preprocess"]:
      if self.file_name in cfg["preprocess"]["save_extra_columns"].keys():
        density_loop += ["Extra"]
        regression_loop += ["Extra"]
        classifier_loop += ["Extra"]

    # Check if we need to split density models
    split_density_model = GetSplitDensityModel(cfg, self.file_name, category=category)

    for data_split in ["train","test"]:
      for loop_value in GetModelLoop(cfg, model_file_name=self.file_name, only_density=True, specific_category=category):

        if not split_density_model:
          density_dir = "density"
        else:
          density_dir = f"density/split_{loop_value['split']}"

        tmp_files = []
        for k in density_loop:
          outfile = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/{density_dir}/{k}_{data_split}.parquet"
          tmp_files.append(outfile)
          if not skip_copy: os.system(f"cp {outfile} {outfile.replace('.parquet','_copy.parquet')}")
        if len(tmp_files) > 0:
          files["density"].append(tmp_files)
          shift_files["density"].append(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/{density_dir}/wt_{data_split}.parquet")
          shift_columns["density"].append("wt")

      for loop_value in GetModelLoop(cfg, model_file_name=self.file_name, only_regression=True, specific_category=category):
        value = cfg["models"][self.file_name]["regression_models"][loop_value["loop_index"]]
        tmp_files = []
        for k in regression_loop:
          outfile = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/regression/{value['parameter']}/{k}_{data_split}.parquet"
          tmp_files.append(outfile)
          if not skip_copy: os.system(f"cp {outfile} {outfile.replace('.parquet','_copy.parquet')}")
        if len(tmp_files) > 0:
          files["regression"].append(tmp_files)
          shift_files["regression"].append(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/regression/{value['parameter']}/wt_{data_split}.parquet")
          shift_columns["regression"].append("wt")

      for loop_value in GetModelLoop(cfg, model_file_name=self.file_name, only_classification=True, specific_category=category):
        value = cfg["models"][self.file_name]["classifier_models"][loop_value["loop_index"]]
        tmp_files = []
        for k in classifier_loop:
          outfile = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/classifier/{value['parameter']}/{k}_{data_split}.parquet"
          tmp_files.append(outfile)
          if not skip_copy: os.system(f"cp {outfile} {outfile.replace('.parquet','_copy.parquet')}")
        if len(tmp_files) > 0:
          files["classifier"].append(tmp_files)
          shift_files["classifier"].append(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/classifier/{value['parameter']}/wt_{data_split}.parquet")
          shift_columns["classifier"].append("wt")

    for data_split in ["val","train_inf","test_inf","full"]:

      # Do validation loop
      for ind in range(len(GetValidationLoop(cfg, self.file_name))):
        tmp_files = []
        for k in density_loop:
          outfile = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/val_ind_{ind}/{k}_{data_split}.parquet"
          tmp_files.append(outfile)
          if not skip_copy: os.system(f"cp {outfile} {outfile.replace('.parquet','_copy.parquet')}")
        if len(tmp_files) > 0:
          files["validation"].append(tmp_files)
          shift_files["validation"].append(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/val_ind_{ind}/wt_{data_split}.parquet")
          shift_columns["validation"].append("wt")

      # Do nuisance variation loop
      for nuisance in [nui for nui in cfg["nuisances"] if nui in GetParametersInModel(self.file_name, cfg, category=category)]:
        for shift in ["up", "down"]:
          tmp_files = []
          for k in density_loop:
            outfile = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/{nuisance}_{shift}/{k}_{data_split}.parquet"
            tmp_files.append(outfile)
            if not skip_copy: os.system(f"cp {outfile} {outfile.replace('.parquet','_copy.parquet')}")
          if len(tmp_files) > 0:
            files["validation"].append(tmp_files)
            shift_files["validation"].append(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/{nuisance}_{shift}/wt_{data_split}.parquet")
            shift_columns["validation"].append("wt")

    return files, shift_files, shift_columns


  def _PlotReweighting(self, normalised_fractions, base_file, wt_func, selection=None, extra_name=None):
    """
    Plot the reweighting of the samples.
    Args:
        normalised_fractions (dict): The normalised fractions to plot.
        base_file (str): The base file to use.
        wt_func (str): The weight function to use.
    """

    print("- Plotting optimally reweighted samples")

    # full dataprocessor
    dp = DataProcessor(
      [[base_file]],
      "parquet",
      batch_size=self.batch_size,
      options = {
        "wt_name" : "wt",
        "selection" : selection
      }
    )

    # calculate weight
    def apply_wt(df, wt_func):
      if self.bw_mass_name not in df.columns:
        df.loc[:,self.bw_mass_name] = 172.5*np.ones(len(df))
      df.loc[:,"wt"] = df.eval(wt_func)
      return df
    apply_wt_partial = partial(apply_wt, wt_func=wt_func)

    unique = dp.GetFull(method="unique")

    for col in self.plot_columns:
      
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
        ignore_quantile=0.02,
        functions_to_apply = [apply_wt_partial]
      )

      colour_list = sns.color_palette("Set2", len(unique[self.mass_name]))

      for i, mass in enumerate(unique[self.mass_name]):

        var_hist, var_hist_uncert, _ = dp.GetFull(
          method="histogram_and_uncert", 
          extra_sel=f"{self.mass_name}=={mass}",
          bins=bins,
          column=col,
          functions_to_apply = [apply_wt_partial]
        )

        integral = np.sum(var_hist*(bins[1]-bins[0]))
        var_hist /= integral
        var_hist_uncert /= integral
        hists.append(var_hist)
        hist_uncerts.append(var_hist_uncert)
        if i == 0:
          hist_names.append(r"Nominal ($m_{t}$ = " + f"{mass} GeV)")
        else:
          hist_names.append(r"$m_{t}$ = " + f"{mass} GeV")
        drawstyles.append("steps-mid")
        colours.append(colour_list[i])

        bw_reweighted_hist, bw_reweighted_hist_uncert, _ = dp.GetFull(
          method="histogram_and_uncert", 
          bins=bins,
          column=col,
          functions_to_apply=[
            apply_wt_partial,
            partial(self._ApplyBWReweight,m=mass,l=self._TopQuarkWidth(mass)),partial(self._ApplyFractions, fractions=normalised_fractions[mass])
          ]
        )

        integral = np.sum(bw_reweighted_hist*(bins[1]-bins[0]))
        bw_reweighted_hist /= integral
        bw_reweighted_hist_uncert /= integral
        error_bar_hists.append(bw_reweighted_hist)
        error_bar_hist_uncerts.append(bw_reweighted_hist_uncert)
        if i == 0:
          error_bar_hist_names.append(r"BW ($m_{t}$ = " + f"{mass} GeV)")
        else:
          error_bar_hist_names.append(None)

      MakeDirectories(self.plot_dir)

      if extra_name is not None:
        plot_name = f"{self.plot_dir}/bw_reweighted_{col}_{extra_name}"
      else:
        plot_name = f"{self.plot_dir}/bw_reweighted_{col}"


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
        name=plot_name, 
        x_label=col, 
        y_label="Density"
      )

      plot_histograms_with_ratio(
        [[error_bar_hists[ind], hists[ind]] for ind in range(len(hists))[::-1]],
        [[error_bar_hist_uncerts[ind], hist_uncerts[ind]] for ind in range(len(hists))[::-1]],
        [[error_bar_hist_names[ind], hist_names[ind]] for ind in range(len(hists))[::-1]],
        np.array(bins),
        xlabel = col,
        ylabel = "Density",
        name = plot_name + "_ratio",
        ratio_range = [0.9,1.1],
        draw_error_bars = True,
      )


  def _TopQuarkWidth(self, m):
    """
    Calculate the width of the top quark.
    Args:
        m (float): The mass of the top quark.
    Returns:
        float: The width of the top quark.
    """
    # Calculate the width of the top quark
    return (0.0270*m) - 3.3455


  def _WriteDataset(self, df, file_name, columns=None):
    """
    Write the dataset to a file.
    Args:
        df (pd.DataFrame): The input dataframe.
        file_name (str): The name of the file to write to.
    """
    # Write the dataset to a file
    file_path = f"{file_name}"
    MakeDirectories(file_path)
    table = pa.Table.from_pandas(df.loc[:,columns] if columns is not None else df, preserve_index=False)
    if os.path.isfile(file_path):
      combined_table = pa.concat_tables([pq.read_table(file_path), table])
      pq.write_table(combined_table, file_path, compression='snappy')
    else:
      pq.write_table(table, file_path, compression='snappy')

    return df


  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    self.write = True if "write" not in self.options else self.options["write"].strip() == "True"
    self.plot = False if "plot" not in self.options else self.options["plot"].strip() == "True"
    self.base_file_name = "base_ttbar_$CATEGORY" if "base_file_name" not in self.options else self.options["base_file_name"]
    self.transformed_to_masses = [166.5,167.5,168.5,169.5,170.5,171.5,172.5,173.5,174.5,175.5,176.5,177.5,178.5]
    #self.transformed_to_masses = [166.5,169.5,171.5,172.5,173.5,175.5,178.5]
    self.use_copies = False if "use_copies" not in self.options else self.options["use_copies"].strip() == "True"
    self.gen_mass = "mass_had_top" if "gen_mass" not in self.options else self.options["gen_mass"].strip()
    self.gen_mass_other = None if "gen_mass_other" not in self.options else self.options["gen_mass_other"].strip()
    self.file_name = "ttbar" if "file_name" not in self.options else self.options["file_name"].strip()
    self.mass_name = "mass" if "mass_name" not in self.options else self.options["mass_name"].strip()
    self.bw_mass_name = "bw_mass" if "bw_mass_name" not in self.options else self.options["bw_mass_name"].strip()
    self.ignore_quantile = 0.1 if "ignore_quantile" not in self.options else float(self.options["ignore_quantile"])

    cfg = LoadConfig(self.cfg)
    #self.base_file = f"{data_dir}/{cfg['name']}/LoadData/{self.base_file_name}.parquet"
    self.plot_columns = [self.gen_mass] + cfg["variables"]
    if self.gen_mass_other is not None:
      self.plot_columns.append(self.gen_mass_other)
    self.plot_dir = f"{plots_dir}/{cfg['name']}/top_bw_fractioning/{self.file_name}"


  def Run(self):
    """
    Run the class.
    """
    # Load the config
    cfg = LoadConfig(self.cfg)

    for category in GetCategoryLoop(cfg):

      base_file_name = self.base_file_name.replace("$CATEGORY",category)
      base_file = f"{data_dir}/{cfg['name']}/LoadData/{base_file_name}.parquet"

      # Calculate optimal fractions
      normalised_fractions, splines = self._CalculateOptimalFractions(base_file, cfg["files"][base_file_name]["weight"], selection=cfg["categories"][category])

      # Plot reweighting
      if self.plot:
        self._PlotReweighting(normalised_fractions, base_file, cfg["files"][base_file_name]["weight"], selection=cfg["categories"][category], extra_name=category)

      # Open parameters
      parameters_name = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/parameters.yaml"
      with open(parameters_name, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Get files
      files, shift_files, shift_columns = self._GetFiles(cfg, category)

      for k in files.keys():

        for ind, splits_per_file in enumerate(files[k]):

          # Use copies or make copies
          for file in splits_per_file:
            copy_name = file.replace(".parquet", "_copy.parquet")
            if self.use_copies:
              os.system(f"cp {copy_name} {file}")
            else:
              os.system(f"cp {file} {copy_name}")

          # Apply weights
          self._DoReweighting(splines, splits_per_file, parameters, k, shift_files[k][ind], shift_columns[k][ind])


      # Get effective events of validation files
      for data_split in ["val","train_inf","test_inf","full"]:
        for ind in range(len(GetValidationLoop(cfg, self.file_name))):
          outfile = [f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/val_ind_{ind}/{k}_{data_split}.parquet" for k in ["X","Y","wt"]]
          dp = DataProcessor(
            [outfile],
            "parquet",
            batch_size=self.batch_size,
            options = {
              "wt_name" : "wt"
            }
          )
          parameters["eff_events"][data_split][ind] = dp.GetFull(method="n_eff")

          # normalise in validation file
          copy_outfile = [f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/val_ind_{ind}/{k}_{data_split}_copy.parquet" for k in ["X","Y","wt"]]
          copy_dp = DataProcessor(
            [copy_outfile],
            "parquet",
            batch_size=self.batch_size,
            options = {
              "wt_name" : "wt"
            }
          )
          prev_sum_wt = copy_dp.GetFull(method="sum")
          sum_wt = dp.GetFull(method="sum")
          infile = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/val_ind_{ind}/wt_{data_split}.parquet"
          outfile = infile.replace('.parquet','_normalised.parquet')
          if os.path.isfile(outfile): os.system(f"rm {outfile}")
          def normalise(df, sum_wt):
            if self.bw_mass_name not in df.columns:
              df.loc[:,self.bw_mass_name] = 172.5
            df.loc[:,"wt"] /= sum_wt
            return df.loc[:,["wt"]]
          dp.GetFull(
            method = None,
            functions_to_apply = [
              partial(normalise, sum_wt=sum_wt/prev_sum_wt),
              partial(self._WriteDataset, file_name=outfile)
            ]
          )
          if os.path.isfile(outfile): os.system(f"mv {outfile} {infile}")    


      # Normalise nuisance variations in validation files
      for data_split in ["val","train_inf","test_inf","full"]:
        for nuisance in [nui for nui in cfg["nuisances"] if nui in GetParametersInModel(self.file_name, cfg, category=category)]:
          for shift in ["up", "down"]:
            outfile = [f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/{nuisance}_{shift}/{k}_{data_split}.parquet" for k in ["X","Y","wt"]]
            dp = DataProcessor(
              [outfile],
              "parquet",
              batch_size=self.batch_size,
              options = {
                "wt_name" : "wt"
              }
            )
            parameters["eff_events"][f"{nuisance}_{shift}"][data_split] = dp.GetFull(method="n_eff")

            # normalise in nuisance variation file
            copy_outfile = [f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/{nuisance}_{shift}/{k}_{data_split}_copy.parquet" for k in ["X","Y","wt"]]
            copy_dp = DataProcessor(
              [copy_outfile],
              "parquet",
              batch_size=self.batch_size,
              options = {
                "wt_name" : "wt"
              }
            )
            prev_sum_wt = copy_dp.GetFull(method="sum")
            sum_wt = dp.GetFull(method="sum")
            infile = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/{nuisance}_{shift}/wt_{data_split}.parquet"
            outfile = infile.replace('.parquet','_normalised.parquet')
            if os.path.isfile(outfile): os.system(f"rm {outfile}")
            def normalise(df, sum_wt):
              if self.bw_mass_name not in df.columns:
                df.loc[:,self.bw_mass_name] = 172.5
              df.loc[:,"wt"] /= sum_wt
              return df.loc[:,["wt"]]
            dp.GetFull(
              method = None,
              functions_to_apply = [
                partial(normalise, sum_wt=sum_wt/prev_sum_wt),
                partial(self._WriteDataset, file_name=outfile)
              ]
            )
            if os.path.isfile(outfile): os.system(f"mv {outfile} {infile}")



      # flatten training bw shapes
      do_density = False
      for loop_value in GetModelLoop(cfg, model_file_name=self.file_name, only_density=True, specific_category=category):
        model = cfg["models"][self.file_name]["density_models"][loop_value["loop_index"]]
        distribution = model["shifts"][self.bw_mass_name] if self.bw_mass_name in model["shifts"].keys() else None
        if self.bw_mass_name in model["parameters"]:
          do_density = True
      if do_density:
        if not GetSplitDensityModel(cfg, self.file_name, category=category):
          self._FlattenTraining(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/density/", ["X","Y","wt","Extra"], parameters["density"], distribution=distribution)
        else:
          for loop_value in GetModelLoop(cfg, model_file_name=self.file_name, only_density=True, specific_category=category):
            ind = loop_value["loop_index"]
            model = cfg["models"][self.file_name]["density_models"][ind]
            if self.bw_mass_name in model["parameters"]:
              self._FlattenTraining(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/density/split_{ind}", ["X","Y","wt","Extra"], parameters["density"][ind], distribution=distribution)
      for loop_value in GetModelLoop(cfg, model_file_name=self.file_name, only_regression=True, specific_category=category):
        model = cfg["models"][self.file_name]["regression_models"][loop_value["loop_index"]]
        if model["parameter"] == self.bw_mass_name:
          self._FlattenTraining(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/regression/{self.bw_mass_name}", ["X","y","wt","Extra"], parameters["regression"][model["parameter"]], distribution=distribution)
      for loop_value in GetModelLoop(cfg, model_file_name=self.file_name, only_classification=True, specific_category=category):
        model = cfg["models"][self.file_name]["classifier_models"][loop_value["loop_index"]]
        if model["parameter"] == self.bw_mass_name:
          self._FlattenTraining(f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/classifier/{self.bw_mass_name}", ["X","y","wt","Extra"], parameters["classifier"][model["parameter"]], distribution=distribution)

      # Change parameters
      if self.write:
        with open(parameters_name, 'w') as yaml_file:
          yaml.dump(parameters, yaml_file)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    for category in GetCategoryLoop(cfg):

      parameters_name = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/parameters.yaml"

      # Add parameters
      outputs += [parameters_name]

      # Get files
      _, shift_files, _ = self._GetFiles(cfg, category, skip_copy=True)
      for k in shift_files.keys():
        for file in shift_files[k]:
          outputs += [file]

      # Add plots
      if self.plot:
        for cat in GetCategoryLoop(cfg):
          for col in self.plot_columns:
            outputs += [f"{self.plot_dir}/bw_reweighted_{col}_{cat}.pdf"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Add config
    inputs = [
      self.cfg,
      ]

    # Load config
    cfg = LoadConfig(self.cfg)

    for category in GetCategoryLoop(cfg):

      base_file_name = self.base_file_name.replace("$CATEGORY",category)
      base_file = f"{data_dir}/{cfg['name']}/LoadData/{base_file_name}.parquet"
      inputs += [base_file]

      parameters_name = f"{data_dir}/{cfg['name']}/PreProcess/{self.file_name}/{category}/parameters.yaml"

      inputs += [parameters_name]

      # Get files
      files, _, _ = self._GetFiles(cfg, category, skip_copy=True)
      for k in files.keys():
        for ind, splits_per_file in enumerate(files[k]):
          for file in splits_per_file:
            inputs += [file]

    return inputs