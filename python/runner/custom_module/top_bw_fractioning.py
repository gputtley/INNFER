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
from useful_functions import MakeDirectories, LoadConfig, GetValidationLoop

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
    df.loc[:,"wt"] *= self._BW(df.loc[:,"mass_had_top"]**2,l=l, m=m)/self._BW(df.loc[:,"mass_had_top"]**2,l=self._TopQuarkWidth(df.loc[:,"mass"]), m=df.loc[:,"mass"]) 
    return df

  def _ApplyFractions(self, df, fractions={}):
    """
    Apply the fractions to the dataframe.
    Args:
        df (pd.DataFrame): The input dataframe.
        fractions (dict): The fractions to apply.
    """
    # Apply the fractions
    df.loc[:,"wt"] *= df.loc[:,"mass"].map(fractions)
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

  def _CalculateOptimalFractions(self, base_file, wt_func):
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
        "wt_name" : "wt"
      }
    )

    # calculate weight
    def apply_wt(df, wt_func):
      df.loc[:,"wt"] = df.eval(wt_func)
      return df
    apply_wt_partial = partial(apply_wt, wt_func=wt_func)

    # get unique Y and loop through
    unique = dp.GetFull(method="unique")["mass"]

    # get nominal sum of weights
    nominal_sum_wt = []
    for transformed_from in unique:
      nominal_sum_wt.append(dp.GetFull(
          method = "sum", 
          extra_sel = f"mass=={transformed_from}", 
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
          extra_sel = f"mass=={transformed_from}", 
          functions_to_apply = [
            apply_wt_partial,
            partial(self._ApplyBWReweight,m=transformed_to,l=self._TopQuarkWidth(transformed_to))
          ]
        )
        sum_wt_squared[transformed_to][transformed_from] = dp.GetFull(
          method = "sum_w2", 
          extra_sel = f"mass=={transformed_from}", 
          functions_to_apply = [
            apply_wt_partial,
            partial(self._ApplyBWReweight,m=transformed_to,l=self._TopQuarkWidth(transformed_to))
          ]
        )

    # derive fractions
    fractions = {}
    for transformed_to in self.transformed_to_masses:
      fractions[transformed_to] = {}
      for transformed_from in unique:
        fractions[transformed_to][transformed_from] = (sum_wt[transformed_to][transformed_from] / sum_wt_squared[transformed_to][transformed_from])


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
    elif file_type == "regression":
      parameters_dict = parameters["regression"]["bw_mass"]
      functions = ["untransform"]
    elif file_type == "validation":
      parameters_dict = {}
      functions = []

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

    wt_file = copy.deepcopy(shift_file)
    wt_alt = shift_file.replace(".parquet","_bw_fractioning.parquet")
    if os.path.isfile(wt_alt):
      os.system(f"rm {wt_alt}")

    def ApplyFractions(df, fraction_splines={}, shift_column="wt"):
      for k, v in fraction_splines.items():
        indices = (df.loc[:,"mass"]==k)
        df.loc[indices,shift_column] *= v(df.loc[indices,"bw_mass"].to_numpy())
      return df.loc[:,[shift_column]]

    dp.GetFull(
      method=None, 
      functions_to_apply = [
        partial(ApplyFractions, fraction_splines=splines, shift_column=shift_column),
        partial(self._WriteDataset, file_name=wt_alt)
      ]
    )

    if self.write:
      os.system(f"mv {wt_alt} {wt_file}")

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
    hist, bins = dp.GetFull(method="histogram", column="bw_mass", bins=40, ignore_quantile=0.0)
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
        for ind in range(len(hist)):
          df.loc[((df.loc[:,"bw_mass"]>=bins[ind]) & (df.loc[:,"bw_mass"]<bins[ind+1])),"wt"] /= hist[ind]     
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

  def _GetFiles(self, cfg):
    """
    Get the files to be used for the reweighting.
    Args:
        cfg (dict): The configuration dictionary.
    Returns:
        tuple: A tuple containing the files, shift files, and shift columns.
    """

    files = {"density":[], "regression":[], "validation":[]}
    shift_files = {"density":[], "regression":[], "validation":[]}
    shift_columns = {"density":[], "regression":[], "validation":[]}


    for data_split in ["train","test"]:
      tmp_files = []
      for k in ["X","Y","wt","Extra"]:
        outfile = f"data/{cfg['name']}/PreProcess/ttbar/density/{k}_{data_split}.parquet"
        if os.path.isfile(outfile):
          tmp_files.append(outfile)
          os.system(f"cp {outfile} {outfile.replace('.parquet','_copy.parquet')}")
      if len(tmp_files) > 0:
        files["density"].append(tmp_files)
        shift_files["density"].append(f"data/{cfg['name']}/PreProcess/ttbar/density/wt_{data_split}.parquet")
        shift_columns["density"].append("wt")

      tmp_files = []
      for k in ["X","y","wt","Extra"]:
        for value in cfg["models"]["ttbar"]["regression_models"]:
          if value["parameter"] == "bw_mass":
            outfile = f"data/{cfg['name']}/PreProcess/ttbar/regression/{value['parameter']}/{k}_{data_split}.parquet"
            if os.path.isfile(outfile):
              tmp_files.append(outfile)
      if len(tmp_files) > 0:
        files["regression"].append(tmp_files)
        shift_files["regression"].append(f"data/{cfg['name']}/PreProcess/ttbar/regression/{value['parameter']}/wt_{data_split}.parquet")
        shift_columns["regression"].append("wt")

    for data_split in ["val","train_inf","test_inf","full"]:
      for ind in range(len(GetValidationLoop(cfg, "ttbar"))):
        tmp_files = []
        for k in ["X","Y","wt","Extra"]:
          outfile = f"data/{cfg['name']}/PreProcess/ttbar/val_ind_{ind}/{k}_{data_split}.parquet"
          if os.path.isfile(outfile):
            tmp_files.append(outfile)
        if len(tmp_files) > 0:
          files["validation"].append(tmp_files)
          shift_files["validation"].append(f"data/{cfg['name']}/PreProcess/ttbar/val_ind_{ind}/wt_{data_split}.parquet")
          shift_columns["validation"].append("wt")

    return files, shift_files, shift_columns

  def _PlotReweighting(self, normalised_fractions, base_file, wt_func, cfg):
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
        "wt_name" : "wt"
      }
    )

    # calculate weight
    def apply_wt(df, wt_func):
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

      colour_list = sns.color_palette("Set2", len(unique["mass"]))

      for i, mass in enumerate(unique["mass"]):

        var_hist, var_hist_uncert, _ = dp.GetFull(
          method="histogram_and_uncert", 
          extra_sel=f"mass=={mass}",
          bins=bins,
          column=col,
          functions_to_apply = [apply_wt_partial]
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
        error_bar_hist_names.append(None)

      MakeDirectories(self.plot_dir)

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
        name=f"{self.plot_dir}/bw_reweighted_{col}", 
        x_label=col, 
        y_label="Density"
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


  def _WriteDataset(self, df, file_name):
    """
    Write the dataset to a file.
    Args:
        df (pd.DataFrame): The input dataframe.
        file_name (str): The name of the file to write to.
    """
    # Write the dataset to a file
    file_path = f"{file_name}"
    MakeDirectories(file_path)
    table = pa.Table.from_pandas(df, preserve_index=False)
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
    self.base_file_name = "base_ttbar" if "base_file_name" not in self.options else self.options["base_file_name"]
    #self.transformed_to_masses = [166.5,167.5,168.5,169.5,170.5,171.5,172.5,173.5,174.5,175.5,176.5,177.5,178.5]
    self.transformed_to_masses = [166.5,169.5,171.5,172.5,173.5,175.5,178.5]
    self.use_copies = False if "use_copies" not in self.options else self.options["use_copies"].strip() == "True"

    cfg = LoadConfig(self.cfg)
    self.parameters_name = f"data/{cfg['name']}/PreProcess/ttbar/parameters.yaml"
    self.base_file = f"data/{cfg['name']}/LoadData/{self.base_file_name}.parquet"
    self.plot_columns = ["mass_had_top"]
    self.plot_dir = f"plots/{cfg['name']}/top_bw_fractioning"

  def Run(self):
    """
    Run the class.
    """
    # Load the config
    cfg = LoadConfig(self.cfg)

    # Open parameters
    with open(self.parameters_name, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Calculate optimal fractions
    normalised_fractions, splines = self._CalculateOptimalFractions(self.base_file, cfg["files"][self.base_file_name]["weight"])

    # Plot reweighting
    self._PlotReweighting(normalised_fractions, self.base_file, cfg["files"][self.base_file_name]["weight"], cfg)

    # Get files
    files, shift_files, shift_columns = self._GetFiles(cfg)

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
      for ind in range(len(GetValidationLoop(cfg, "ttbar"))):
        outfile = [f"data/{cfg['name']}/PreProcess/ttbar/val_ind_{ind}/{k}_{data_split}.parquet" for k in ["X","Y","wt"]]
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
        copy_outfile = [f"data/{cfg['name']}/PreProcess/ttbar/val_ind_{ind}/{k}_{data_split}_copy.parquet" for k in ["X","Y","wt"]]
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
        infile = f"data/{cfg['name']}/PreProcess/ttbar/val_ind_{ind}/wt_{data_split}.parquet"
        outfile = infile.replace('.parquet','_normalised.parquet')
        if os.path.isfile(outfile): os.system(f"rm {outfile}")
        def normalise(df, sum_wt):
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
    for model in cfg["models"]["ttbar"]["density_models"]:
      if "bw_mass" in model["parameters"]:
        do_density = True
    if do_density:
      self._FlattenTraining(f"data/{cfg['name']}/PreProcess/ttbar/density/", ["X","Y","wt","Extra"])
    for model in cfg["models"]["ttbar"]["regression_models"]:
      if model["parameter"] == "bw_mass":
        self._FlattenTraining(f"data/{cfg['name']}/PreProcess/ttbar/regression/bw_mass", ["X","y","wt","Extra"])

    # Change parameters
    if self.write:
      with open(self.parameters_name, 'w') as yaml_file:
        yaml.dump(parameters, yaml_file)

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add parameters
    outputs += [self.parameters_name]

    # Get files
    _, shift_files, _ = self._GetFiles(cfg)
    for k in shift_files.keys():
      for file in shift_files[k]:
        outputs += [file]

    # Add plots
    for col in self.plot_columns:
      outputs += [f"{self.plot_dir}/bw_reweighted_{col}.pdf"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Add config
    inputs = [
      self.cfg,
      self.parameters_name,
      self.base_file,        
      ]

    # Load config
    cfg = LoadConfig(self.cfg)

    # Get files
    files, _, _ = self._GetFiles(cfg)
    for k in files.keys():
      for ind, splits_per_file in enumerate(files[k]):
        for file in splits_per_file:
          inputs += [file]

    return inputs