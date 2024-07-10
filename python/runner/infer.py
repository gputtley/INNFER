import copy
import os
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from plotting import plot_stacked_histogram_with_ratio
from useful_functions import MakeDirectories

class Infer():

  def __init__(self):

    self.parameters =  None
    self.model = None
    self.architecture = None

    self.true_Y = None
    self.initial_best_fit_guess = None
    self.pois = None
    self.nuisances = None
    self.method = "InitialFit"

    self.inference_options = {}
    self.yield_function = "default",
    self.data_type = "sim"
    self.data_input = "data"
    self.data_output = "data"
    self.plot_output = "plots"
    self.likelihood_type = "unbinned_extended"
    self.resample = False
    self.resampling_seed = 1
    self.verbose = True
    self.n_asimov_events = 10**6
    self.minimisation_method = "nominal"
    self.freeze = {}
    self.extra_file_name = ""
    self.scale_to_eff_events = False
    self.column = None
    self.scan_value = None
    self.scan_ind = None
    self.sigma_between_scan_points = 0.4
    self.number_of_scan_points = 17
    self.other_input = None
    self.other_input_files = []
    self.other_output_files = []
    self.model_type = "BayesFlow"
    self.data_file = None
    self.write_asimov = True
    self.asimov_input = None
    self.plot_resampling = True
    self.binned_fit_input = None
    self.yields = None
    self.dps = None

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    # Make singular inputs as dictionaries
    combined_model = True
    if isinstance(self.model, str):
      combined_model = False
      with open(self.parameters, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
      self.model = {parameters['file_name'] : self.model}
      self.parameters = {parameters['file_name'] : self.parameters}
      self.architecture = {parameters['file_name'] : self.architecture}

    if self.extra_file_name != "":
      self.extra_file_name = f"_{self.extra_file_name}"


  def Run(self):

    # Change datatype
    if self.likelihood_type in ["binned", "binned_extended"] and self.data_type == "asimov":
      self.data_type = "sim"

    # Make yields
    self.yields = self._BuildYieldFunctions()

    # Make data processors
    self.dps = self._BuildDataProcessors()

    # Make likelihood
    lkld = self._BuildLikelihood()

    # Make likelihood inputs
    if self.likelihood_type in ["unbinned", "unbinned_extended"]:
      lkld_input = self.dps.values()
    elif self.likelihood_type in ["binned", "binned_extended"]:
      lkld_input = None

    # Method to make the asimov datasets
    if self.method == "MakeAsimov" and self.likelihood_type in ["unbinned", "unbinned_extended"]:

      import tensorflow as tf
      from data_processor import DataProcessor

      sum_yields = np.sum([lkld.models["yields"][fn](self.true_Y) for fn in self.parameters.keys()])

      for file_name in self.parameters.keys():
        if lkld.models["yields"][file_name](self.true_Y) == 0: continue
        frac_yields = lkld.models["yields"][file_name](self.true_Y)/sum_yields
        asimov_writer = DataProcessor(
          [[partial(lkld.models["pdfs"][file_name].Sample, self.true_Y)]],
          "generator",
          n_events = int(self.n_asimov_events*frac_yields),
          wt_name = "wt",
          options = {
            "parameters" : lkld.data_parameters[file_name],
            "scale" : lkld.models["yields"][file_name](self.true_Y),
          }
        )
        if self.verbose:
          print(f"- Making asimov dataset for {file_name} and writing it to file")
        asimov_file_name = f"{self.data_output}/asimov_{file_name}{self.extra_file_name}.parquet"
        MakeDirectories(asimov_file_name)
        if os.path.isfile(asimov_file_name): os.system(f"rm {asimov_file_name}")
        tf.random.set_seed(lkld.seed)
        tf.keras.utils.set_random_seed(lkld.seed)
        asimov_writer.GetFull(
          method = None,
          functions_to_apply = [
            partial(self._WriteDatasets,file_path=asimov_file_name)
          ]
        )
      return None

    if self.verbose:
      if self.data_type != "data":
        print(f"- Performing actions on the likelihood for the dataframe on data type {self.data_type}:")
        print(self.true_Y)
      else:
        print(f"- Performing actions on the likelihood on data")

    if self.method == "InitialFit":

      if self.verbose:
        print("- Likelihood output:")

      lkld.GetAndWriteBestFitToYaml(
        lkld_input, 
        self.initial_best_fit_guess, 
        row=self.true_Y, 
        filename=f"{self.data_output}/best_fit{self.extra_file_name}.yaml", 
        minimisation_method=self.minimisation_method, 
        freeze=self.freeze,
      )

    elif self.method == "Hessian":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.data_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      lkld.best_fit = np.array(best_fit_info["best_fit"])
      lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Calculating the Hessian matrix")

      # Get Hessian
      lkld.GetAndWriteHessianToYaml(
        lkld_input, 
        row=self.true_Y, 
        filename=f"{self.data_output}/hessian{self.extra_file_name}.yaml", 
        freeze=self.freeze,
      )      

    elif self.method == "ScanPoints":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.data_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      lkld.best_fit = np.array(best_fit_info["best_fit"])
      lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Finding values to perform the scan")
        print("- Likelihood output:")

      # Make scan ranges
      lkld.GetAndWriteScanRangesToYaml(
        lkld_input, 
        self.column,
        row=self.true_Y,
        estimated_sigmas_shown=((self.number_of_scan_points-1)/2)*self.sigma_between_scan_points, 
        estimated_sigma_step=self.sigma_between_scan_points,
        filename=f"{self.data_output}/scan_ranges_{self.column}{self.extra_file_name}.yaml"
      )

    elif self.method == "Scan":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.data_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      lkld.best_fit = np.array(best_fit_info["best_fit"])
      lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Performing likelihood profiling")
        print(f"- Likelihood output:")

      lkld.GetAndWriteScanToYaml(
        lkld_input, 
        self.column, 
        self.scan_value, 
        row=self.true_Y, 
        freeze=self.freeze, 
        filename=f"{self.data_output}/scan_values_{self.column}{self.extra_file_name}_{self.scan_ind}.yaml", 
        minimisation_method=self.minimisation_method
      )

    elif self.method == "Debug":

      if self.verbose:
        print(f"- Likelihood output:")

      for j in self.other_input.split(":"):
        lkld.Run(lkld_input, [float(i) for i in j.split(',')], Y_columns=list(self.initial_best_fit_guess.columns))

  def Outputs(self):

    # Initiate outputs
    outputs = []

    # Add asimov files
    if self.method == "MakeAsimov":
      yields = self._BuildYieldFunctions()
      for file_name in self.parameters.keys():
        if yields[file_name](self.true_Y) == 0: continue
        outputs += [f"{self.data_output}/asimov_{file_name}{self.extra_file_name}.parquet"]

    # Add best fit
    if self.method == "InitialFit":
      outputs += [f"{self.data_output}/best_fit{self.extra_file_name}.yaml"]

    # Add scan ranges
    if self.method == "ScanPoints":
      outputs += [f"{self.data_output}/scan_ranges_{self.column}{self.extra_file_name}.yaml",]

    # Add scan values
    if self.method == "Scan":
      outputs += [f"{self.data_output}/scan_values_{self.column}{self.extra_file_name}_{self.scan_ind}.yaml",]

    # Add other outputs
    outputs += self.other_output_files

    return outputs

  def Inputs(self):

    # Initiate inputs
    inputs = []

    # Add parameters and model inputs
    for k in self.parameters.keys():
      inputs += [
        self.model[k], # remove this if not needed
        self.architecture[k],
        self.parameters[k],
      ]

    # Add input datasets
    if self.data_type == "sim":
      for k, v in self.parameters.items():
        with open(v, 'r') as yaml_file:
          parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
        inputs += [
          f"{parameters['file_loc']}/X_val.parquet",
          f"{parameters['file_loc']}/Y_val.parquet",
          f"{parameters['file_loc']}/wt_val.parquet",
        ]
    elif self.data_type == "data":
      inputs += [self.data_file]      
    elif self.data_type == "asimov" and self.write_asimov and self.method != "MakeAsimov":
      for file_name in self.parameters.keys():
        inputs += [f"{self.asimov_input}/asimov_{file_name}{self.extra_file_name}.parquet"]

    # Add best fit if Scan or ScanPoints
    if self.method in ["ScanPoints","Scan"]:
      inputs += [f"{self.data_input}/best_fit{self.extra_file_name}.yaml"]

    # Add other inputs
    inputs += self.other_input_files

    return inputs

  def _BuildCategories(self):

    # Make selection, variable and bins
    categories = {}
    for category_ind, category in enumerate(self.binned_fit_input.split(";")):

      # Make selection, variable and bins
      if ":" in category:
        sel = category.split(":")[0]
        var_and_bins = category.split(":")[1]
      else:
        sel = None
        var_and_bins = category
      var = var_and_bins.split("[")[0]
      bins = [float(i) for i in list(var_and_bins.split("[")[1].split("]")[0].split(","))]
      categories[category_ind] = [sel, var, bins]
    
    return categories

  def _BuildBinYields(self):

    from data_processor import DataProcessor

    # Make selection, variable and bins
    categories = self._BuildCategories()

    # Loop through files
    for file_name, v in self.parameters.items():

      # Open data parameters
      with open(v, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Loop through Y values - will do this from the yields file  
      yields_df = pd.read_parquet(parameters['yield_loc'])
      Y_columns = [i for i in self.pois+self.nuisances if i in yields_df.columns]
      yields_df = yields_df.loc[:,Y_columns]
      for index, row in yields_df.iterrows():
        y_selection = " & ".join([f"({k}=={row.iloc[ind]})" for ind, k in enumerate(row.keys())])
        if y_selection == "": y_selection = None

        # Make data processor
        asimov_dp = DataProcessor(
          [
            [f"{parameters['file_loc']}/X_train.parquet", f"{parameters['file_loc']}/Y_train.parquet", f"{parameters['file_loc']}/wt_train.parquet"],
            [f"{parameters['file_loc']}/X_test.parquet", f"{parameters['file_loc']}/Y_test.parquet", f"{parameters['file_loc']}/wt_test.parquet"],
            [f"{parameters['file_loc']}/X_val.parquet", f"{parameters['file_loc']}/Y_val.parquet", f"{parameters['file_loc']}/wt_val.parquet"],
          ],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : parameters,
            "selection" : y_selection,
            "scale" : self.yields[file_name](yields_df.iloc[[index]]) if self.yields is not None else 1.0,
            "functions" : ["untransform"]
          }
        )

        # Loop through categories
        for cat_num, cat_info in categories.items():

          # Make histograms
          hist, _ = asimov_dp.GetFull(
            method = "histogram",
            extra_sel = cat_info[0],
            column = cat_info[1],
            bins = cat_info[2],
          )

          print(file_name, y_selection, cat_num, cat_info)
          print(hist)
          print()
          # Loop through histogram bins

          # Fill yield dataframes

    exit()

  def _BuildDataProcessors(self):

    from data_processor import DataProcessor

    dps = {}
    for file_name, v in self.parameters.items():

      # Open data parameters
      with open(v, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Generate simulated data processor
      if self.data_type == "sim":

        if self.yields is not None:
          scale = self.yields[file_name](self.true_Y)
          if scale == 0: continue
        else:
          scale = None

        shape_Y_cols = [col for col in self.true_Y.columns if "mu_" not in col and col in parameters["Y_columns"]]
        dps[file_name] = DataProcessor(
          [[f"{parameters['file_loc']}/X_val.parquet", f"{parameters['file_loc']}/Y_val.parquet", f"{parameters['file_loc']}/wt_val.parquet"]],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : parameters,
            "selection" : " & ".join([f"({col}=={self.true_Y.loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
            "scale" : scale,
            "resample" : self.resample,
            "resampling_seed" : self.resampling_seed,
            "functions" : ["untransform"]
          }
        )

        # plot resampling
        if self.plot_resampling and self.resample:
          non_resampling_dp = DataProcessor(
            [[f"{parameters['file_loc']}/X_val.parquet", f"{parameters['file_loc']}/Y_val.parquet", f"{parameters['file_loc']}/wt_val.parquet"]],
            "parquet",
            wt_name = "wt",
            options = {
              "parameters" : parameters,
              "selection" : " & ".join([f"({col}=={self.true_Y.loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
              "scale" : scale,
              "functions" : ["untransform"]
            }
          )
          for col in parameters["X_columns"]:
            # Get binning
            bins = non_resampling_dp.GetFull(
              method = "bins_with_equal_spacing", 
              bins = 40,
              column = col,
            )      
            # Get histograms
            non_resampled_hist, non_resampled_hist_hist_uncert, bins = non_resampling_dp.GetFull(
              method = "histogram_and_uncert",
              bins = bins,
              column = col,
            )
            resampled_hist, resampled_hist_hist_uncert, bins =  dps[file_name].GetFull(
              method = "histogram_and_uncert",
              bins = bins,
              column = col,
            )
            plot_stacked_histogram_with_ratio(
              resampled_hist, 
              {"Full Dataset" : non_resampled_hist}, 
              bins, 
              data_name = f"Resampled with Seed {self.resampling_seed}", 
              xlabel=col,
              ylabel="Events",
              name=f"{self.plots_output}/resampling_{col}_seed_{self.resampling_seed}", 
              data_errors=resampled_hist_hist_uncert, 
              stack_hist_errors=non_resampled_hist_hist_uncert, 
              title_right="",
              use_stat_err=False,
              axis_text="",
              )

      elif self.data_type == "asimov":

        # Generate asimov as you go
        if not self.write_asimov:

          if self.yields is not None:
            sum_yields = np.sum([self.yields[fn](self.true_Y) for fn in self.parameters.keys()])
            scale = self.yields[file_name](self.true_Y)
            if scale == 0.0: continue
            frac_yields = scale/sum_yields
          else:
            frac_yields = 1.0
            scale = None

          dps[file_name] = DataProcessor(
            [[partial(self.models[file_name].Sample, self.true_Y)]],
            "generator",
            n_events = int(self.n_asimov_events*frac_yields), # multiply by yield fractions
            wt_name = "wt",
            options = {
              "parameters" : parameters,
              "scale" : scale,
            }
          )

        # Use asimov that has been written to file file first 
        else:

          asimov_file_name = f"{self.asimov_input}/asimov_{file_name}{self.extra_file_name}.parquet"
          dps[file_name] = DataProcessor(
            [[asimov_file_name]],
            "parquet",
            wt_name = "wt"
          )

    if self.data_type == "data":

      dps["Data"] = DataProcessor(
        [[self.data_file]],
        "parquet",
      )

    return dps

  def _BuildYieldFunctions(self):

    from yields import Yields

    # Load parameters
    parameters = {}
    for k, v in self.parameters.items():
      with open(v, 'r') as yaml_file:
        parameters[k] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build yield function
    yields = {}
    yields_class = {}
    eff_events_class = {}
    sum_wts_squared = 0.0
    sum_wts = 0.0

    for ind, (k, v) in enumerate(self.parameters.items()):
    
      yields_class[k] = Yields(
        pd.read_parquet(parameters[k]['yield_loc']), 
        self.pois, 
        self.nuisances, 
        k,
        method=self.yield_function, 
        column_name="yield"
      )

      if not self.scale_to_eff_events:
        yields[k] = yields_class[k].GetYield

      else:
        eff_events_class[k] = Yields(
          pd.read_parquet(parameters[k]['yield_loc']), 
          self.pois, 
          self.nuisances, 
          k,
          method=self.yield_function, 
          column_name="effective_events_val"
        )
        sum_wts_squared += (yields_class[k].GetYield(self.true_Y)**2)/eff_events_class[k].GetYield(self.true_Y)
        sum_wts += yields_class[k].GetYield(self.true_Y)

    if self.scale_to_eff_events:
      eff_events = (sum_wts**2)/sum_wts_squared
      for k in self.parameters.keys():
        yields[k] = partial(lambda Y, k, eff_events, sum_wts: yields_class[k].GetYield(Y) * (eff_events/sum_wts), k=k, eff_events=eff_events, sum_wts=sum_wts)

    return yields

  def _BuildModels(self):

    if self.model_type == "BayesFlow":

      from network import Network

      networks = {}
      parameters = {}
      for file_name in self.parameters.keys():

        # Open parameters
        if self.verbose:
          print(f"- Loading in the parameters for model {file_name}")
        with open(self.parameters[file_name], 'r') as yaml_file:
          parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Load the architecture in
        if self.verbose:
          print(f"- Loading in the architecture for model {file_name}")
        with open(self.architecture[file_name], 'r') as yaml_file:
          architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Build model
        if self.verbose:
          print(f"- Building the model for {file_name}")
        networks[file_name] = Network(
          f"{parameters[file_name]['file_loc']}/X_train.parquet",
          f"{parameters[file_name]['file_loc']}/Y_train.parquet", 
          f"{parameters[file_name]['file_loc']}/wt_train.parquet", 
          f"{parameters[file_name]['file_loc']}/X_test.parquet",
          f"{parameters[file_name]['file_loc']}/Y_test.parquet", 
          f"{parameters[file_name]['file_loc']}/wt_test.parquet",
          options = {
            **architecture,
            **{
              "data_parameters" : parameters[file_name]
            }
          }
        )  
        
        # Loading model
        if self.verbose:
          print(f"- Loading the model for {file_name}")
        networks[file_name].Load(name=self.model[file_name])

    elif self.model_type.startswith("Benchmark"):

      if not "Dim1CfgToBenchmark" in self.model_type:
        import importlib
        module_name = self.model_type.split("Benchmark_")[1]
        module = importlib.import_module(module_name)
        module_class = getattr(module, module_name)
        networks = {}
        for file_name in self.parameters.keys():
          networks[file_name] = module_class(file_name=file_name)
      else:
        from Dim1CfgToBenchmark import Dim1CfgToBenchmark
        cfg_name = self.model_type.split("Dim1CfgToBenchmark_")[-1]
        for file_name, params in self.parameters.items():
          networks[file_name] = module_class(cfg_name, params, file_name=file_name)                  

    return networks

  def _BuildLikelihood(self):

    from likelihood import Likelihood

    if self.verbose:
      print(f"- Building likelihood")

    parameters = {}
    for file_name in self.parameters.keys():
      with open(self.parameters[file_name], 'r') as yaml_file:
        parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if self.likelihood_type in ["unbinned", "unbinned_extended"]:
      likelihood_inputs = {
        "pdfs" : self._BuildModels(),
        "yields" : self.yields,
      }
    elif self.likelihood_type in ["binned", "binned_extended"]:
      likelihood_inputs = {
        "bin_yields" : self._BuildBinYields(),
      }

    lkld = Likelihood(
      likelihood_inputs, 
      likelihood_type = self.likelihood_type, 
      data_parameters = parameters,
      parameters = self.inference_options,
    )

    return lkld

  def _WriteDatasets(self, df, file_path="data.parquet"):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(file_path):
      combined_table = pa.concat_tables([pq.read_table(file_path), table])
      pq.write_table(combined_table, file_path, compression='snappy')
    else:
      pq.write_table(table, file_path, compression='snappy')
    return df