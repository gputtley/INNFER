import os
import pickle
import yaml

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from pprint import pprint

from useful_functions import (
  GetBinValues,
  InitiateDensityModel, 
  InitiateRegressionModel, 
  InitiateClassifierModel,
  MakeDirectories
)

class Infer():

  def __init__(self):

    self.parameters = None
    self.model_input = None
    self.density_models = {}
    self.regression_models = {}
    self.classifier_models = {}

    self.true_Y = None
    self.val_ind = None
    self.initial_best_fit_guess = None
    self.method = "InitialFit"

    self.inference_options = {}
    self.yield_function = "default"
    self.data_input = "data"
    self.data_output = "data"
    self.plot_output = "plots"
    self.likelihood_type = "unbinned_extended"
    self.verbose = True
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
    self.binned_fit_input = None
    self.yields = None
    self.dps = None
    self.lkld = None
    self.lkld_input = None
    self.sim_type = "val"
    self.binned_sim_type = "full"
    self.X_columns = None
    self.Y_columns = None
    self.Y_columns_per_model = {}
    self.only_density = False
    self.best_fit_input = None
    self.hessian_input = None
    self.d_matrix_input = None
    self.hessian_parallel_column_1 = None
    self.hessian_parallel_column_2 = None
    self.extra_density_model_name = ""
    self.non_nn_columns = []
    self.binned_fit_morph_col = None
    self.binned_data_input = None

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.extra_file_name != "":
      self.extra_file_name = f"_{self.extra_file_name}"

  def Run(self):

    # Make yields or use existing
    if self.yields is None:
      self.yields = self._BuildYieldFunctions()

    # Make data processors or use existing
    if self.dps is None:
      self.dps = self._BuildDataProcessors()

    # Make likelihood or use exisiting
    if self.lkld is None:
      self.lkld = self._BuildLikelihood()

    # Make likelihood inputs
    if self.lkld_input is None:
      if self.likelihood_type in ["unbinned", "unbinned_extended"]:
        self.lkld_input = {k: v.values() for k, v in self.dps.items()}
      elif self.likelihood_type in ["binned", "binned_extended"]:
        self.lkld_input = self.binned_data_input

    if self.verbose:
      if self.freeze != {}:
        print(f"- Freezing parameters: {self.freeze}")

    if self.method == "InitialFit":

      if self.verbose:
        print("- Likelihood output:")

      self.lkld.GetAndWriteBestFitToYaml(
        self.lkld_input, 
        self.initial_best_fit_guess, 
        row=self.true_Y, 
        filename=f"{self.data_output}/best_fit{self.extra_file_name}.yaml", 
        minimisation_method=self.minimisation_method, 
        freeze=self.freeze,
      )

    elif self.method == "ApproximateUncertainty":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      self.lkld.best_fit = np.array(best_fit_info["best_fit"])
      self.lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Finding approximate uncertainties")
        print("- Likelihood output:")

      # Make scan ranges
      self.lkld.GetAndWriteApproximateUncertaintyToYaml(
        self.lkld_input, 
        self.column,
        row=self.true_Y,
        filename=f"{self.data_output}/approximateuncertainty_results_{self.column}{self.extra_file_name}.yaml"
      )

    elif self.method in ["Hessian","HessianParallel","HessianNumerical"]:

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      self.lkld.best_fit = np.array(best_fit_info["best_fit"])
      self.lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Calculating the Hessian matrix")

      extra_col_name = ""
      if self.method == "HessianParallel":
        extra_col_name = f"_{self.hessian_parallel_column_1}_{self.hessian_parallel_column_2}"

      # Get Hessian
      self.lkld.GetAndWriteHessianToYaml(
        self.lkld_input, 
        row=self.true_Y, 
        filename=f"{self.data_output}/hessian{self.extra_file_name}{extra_col_name}.yaml", 
        freeze=self.freeze,
        specific_column_1=self.hessian_parallel_column_1,
        specific_column_2=self.hessian_parallel_column_2,
        numerical = (self.method == "HessianNumerical"),
      )    

    elif self.method == "HessianCollect":

      first = True
      columns = [i for i in self.Y_columns if self.freeze is None or i not in self.freeze.keys()]

      for column_1_ind, column_1 in enumerate(columns):
        for column_2_ind, column_2 in enumerate(columns):
          if column_1_ind > column_2_ind: continue
          # load in the hessian
          if self.verbose:
            print(f"- Loading hessian for {column_1} and {column_2}")
          with open(f"{self.hessian_input}/hessian{self.extra_file_name}_{column_1}_{column_2}.yaml", 'r') as yaml_file:
            hessian = yaml.load(yaml_file, Loader=yaml.FullLoader)
          if first:
            hessian_metadata = hessian
            total_hessian = np.array(hessian["hessian"])
            first = False
          else:
            total_hessian = np.add(total_hessian, np.array(hessian["hessian"]))
      hessian_metadata["hessian"] = total_hessian.tolist()

      # write out the hessian
      if self.verbose:
        print(f"- Writing out the hessian")
      hess_out_name = f"{self.data_output}/hessian{self.extra_file_name}.yaml"
      MakeDirectories(hess_out_name)
      with open(hess_out_name, 'w') as yaml_file: 
        yaml.dump(hessian_metadata, yaml_file, default_flow_style=False)

    elif self.method == "DMatrix":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      self.lkld.best_fit = np.array(best_fit_info["best_fit"])
      self.lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Calculating the D matrix")

      # Calculate the D matrix
      self.lkld.GetAndWriteDMatrixToYaml(
        self.lkld_input, 
        row=self.true_Y, 
        filename=f"{self.data_output}/dmatrix{self.extra_file_name}.yaml", 
        freeze=self.freeze,
      )

    elif self.method == "Covariance":

      if self.verbose:
        print(f"- Calculating the covariance matrix")
      
      with open(f"{self.hessian_input}/hessian{self.extra_file_name}.yaml", 'r') as yaml_file:
        hessian = yaml.load(yaml_file, Loader=yaml.FullLoader)

      self.lkld.best_fit = np.array(hessian["best_fit"])
      self.lkld.hessian = hessian["hessian"]
      self.lkld.hessian_columns = hessian["matrix_columns"]

      # Get covariance
      self.lkld.GetAndWriteCovarianceToYaml(
        row=self.true_Y, 
        filename=f"{self.data_output}/covariance{self.extra_file_name}.yaml", 
        scan_over=hessian["matrix_columns"],
      )    

      # Get uncertainties from covariance
      for col in hessian["matrix_columns"]:
        self.lkld.GetAndWriteCovarianceIntervalsToYaml(
          col,
          row=self.true_Y, 
          filename=f"{self.data_output}/covariance_results_{col}{self.extra_file_name}.yaml", 
          scan_over=hessian["matrix_columns"],
        )    

    elif self.method == "CovarianceWithDMatrix":

      # Open hessian yaml
      with open(f"{self.hessian_input}/hessian{self.extra_file_name}.yaml", 'r') as yaml_file:
        hessian = yaml.load(yaml_file, Loader=yaml.FullLoader)
      # Open D matrix yaml
      with open(f"{self.d_matrix_input}/dmatrix{self.extra_file_name}.yaml", 'r') as yaml_file:
        Dmatrix = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Save to class
      self.lkld.best_fit = np.array(hessian["best_fit"])
      self.lkld.hessian = hessian["hessian"]
      self.lkld.hessian_columns = hessian["matrix_columns"]
      self.lkld.D_matrix = Dmatrix["D_matrix"]
      self.lkld.D_matrix_columns = Dmatrix["matrix_columns"]

      # Get covariance
      self.lkld.GetAndWriteCovarianceToYaml(
        row=self.true_Y, 
        filename=f"{self.data_output}/covariancewithdmatrix{self.extra_file_name}.yaml", 
        scan_over=hessian["matrix_columns"],
      )    

      with open(f"{self.data_output}/covariancewithdmatrix{self.extra_file_name}.yaml", 'r') as yaml_file:
        covariance = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Get uncertainties from covariance
      for col in covariance["matrix_columns"]:
        col_index = covariance["matrix_columns"].index(col)
        best_fit_col = hessian["best_fit"][hessian["columns"].index(col)]
        dump = {
          "columns" : covariance["columns"],
          "row" : covariance["row"],
          "varied_column" : col,
          "crossings" : {
            -2 : best_fit_col - 2*float(np.sqrt(covariance["covariance"][col_index][col_index])),
            -1 : best_fit_col - float(np.sqrt(covariance["covariance"][col_index][col_index])),
            0 : best_fit_col,
            1 : best_fit_col + float(np.sqrt(covariance["covariance"][col_index][col_index])),
            2 : best_fit_col + 2*float(np.sqrt(covariance["covariance"][col_index][col_index])),
          }
        }
        filename = f"{self.data_output}/covariancewithdmatrix_results_{col}{self.extra_file_name}.yaml"
        if self.verbose:
          pprint(dump)
        print(f"Created {filename}")
        MakeDirectories(filename)
        with open(filename, 'w') as yaml_file:
          yaml.dump(dump, yaml_file, default_flow_style=False)

    elif self.method == "ScanPointsFromApproximate":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      self.lkld.best_fit = np.array(best_fit_info["best_fit"])
      self.lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Finding values to perform the scan")
        print("- Likelihood output:")

      # Make scan ranges
      self.lkld.GetAndWriteScanRangesToYaml(
        self.lkld_input, 
        self.column,
        row=self.true_Y,
        estimated_sigmas_shown=((self.number_of_scan_points-1)/2)*self.sigma_between_scan_points, 
        estimated_sigma_step=self.sigma_between_scan_points,
        filename=f"{self.data_output}/scan_ranges_{self.column}{self.extra_file_name}.yaml",
        method="approximate",
      )

    elif self.method == "ScanPointsFromHessian":

      # Open hessian
      with open(f"{self.hessian_input}/hessian{self.extra_file_name}.yaml", 'r') as yaml_file:
        hessian = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      self.lkld.best_fit = np.array(hessian["best_fit"])
      self.lkld.hessian = hessian["hessian"]

      # Make scan ranges
      self.lkld.GetAndWriteScanRangesToYaml(
        self.lkld_input, 
        self.column,
        row=self.true_Y,
        estimated_sigmas_shown=((self.number_of_scan_points-1)/2)*self.sigma_between_scan_points, 
        estimated_sigma_step=self.sigma_between_scan_points,
        filename=f"{self.data_output}/scan_ranges_{self.column}{self.extra_file_name}.yaml",
        method="hessian",
      )

    elif self.method == "Scan":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      self.lkld.best_fit = np.array(best_fit_info["best_fit"])
      self.lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Performing likelihood profiling")
        print(f"- Likelihood output:")

      self.lkld.GetAndWriteScanToYaml(
        self.lkld_input, 
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
        self.lkld.Run(self.lkld_input, [float(i) for i in j.split(',')])

  def Outputs(self):

    # Initiate outputs
    outputs = []

    # Add best fit
    if self.method == "InitialFit":
      outputs += [f"{self.data_output}/best_fit{self.extra_file_name}.yaml"]

    # Add hessian
    if self.method in ["Hessian","HessianCollect","HessianNumerical"]:
      outputs += [f"{self.data_output}/hessian{self.extra_file_name}.yaml"]

    # Add hessian parallel
    if self.method == "HessianParallel":
      outputs += [f"{self.data_output}/hessian{self.extra_file_name}_{self.hessian_parallel_column_1}_{self.hessian_parallel_column_2}.yaml"]

    # Add dmatrix
    if self.method == "DMatrix":
      outputs += [f"{self.data_output}/dmatrix{self.extra_file_name}.yaml"]

    columns = [i for i in self.Y_columns if self.freeze is None or i not in self.freeze.keys()]

    # Add covariance
    if self.method == "Covariance":
      outputs += [f"{self.data_output}/covariance{self.extra_file_name}.yaml"]
      for col in columns:      
        outputs += [f"{self.data_output}/covariance_results_{col}{self.extra_file_name}.yaml"]

    # Add covariancewithdmatrix
    if self.method == "CovarianceWithDMatrix":
      outputs += [f"{self.data_output}/covariancewithdmatrix{self.extra_file_name}.yaml"]
      for col in columns:      
        outputs += [f"{self.data_output}/covariancewithdmatrix_results_{col}{self.extra_file_name}.yaml"]

    # Add scan ranges
    if self.method in ["ScanPointsFromApproximate","ScanPointsFromHessian"]:
      outputs += [f"{self.data_output}/scan_ranges_{self.column}{self.extra_file_name}.yaml",]

    # Add scan values
    if self.method == "Scan":
      outputs += [f"{self.data_output}/scan_values_{self.column}{self.extra_file_name}_{self.scan_ind}.yaml",]

    # Add approximate uncertainty
    if self.method == "ApproximateUncertainty":
      outputs += [f"{self.data_output}/approximateuncertainty_results_{self.column}{self.extra_file_name}.yaml"]

    # Add other outputs
    outputs += self.other_output_files

    return outputs

  def Inputs(self):

    # Initiate inputs
    inputs = []
  
    # Add parameters
    for cat, par in self.parameters.items():
      for k, v in par.items():
        inputs += [v]

    # Add data inputs and parameters
    if self.likelihood_type in ["unbinned", "unbinned_extended"]:
      for cat, par in self.parameters.items():
        for k, v in par.items():
          if "data" not in self.data_input[cat].keys():
            inputs += self.data_input[cat][k]

    # Add data inputs
    for cat, par in self.parameters.items():
      if "data" in self.data_input[cat].keys():
        inputs += self.data_input[cat]["data"]

    if self.likelihood_type in ["unbinned", "unbinned_extended"]:

      # Add density model inputs
      for cat, models in self.density_models.items():
        for k, v in models.items():
          inputs += [
            f"{self.model_input}/{v['name']}{self.extra_density_model_name}/{k}_architecture.yaml",
            f"{self.model_input}/{v['name']}{self.extra_density_model_name}/{k}.h5",
          ]

      # Add regression model inputs
      for cat, models in self.regression_models.items():
        for k, v in models.items():
          for vi in v:
            inputs += [
              f"{self.model_input}/{vi['name']}/{k}_architecture.yaml",
              f"{self.model_input}/{vi['name']}/{k}.h5",
              f"{self.model_input}/{vi['name']}/{k}_norm_spline.pkl"
            ]

      # Add classifier model inputs
      for cat, models in self.classifier_models.items():
        for k, v in models.items():
          for vi in v:
            inputs += [
              f"{self.model_input}/{vi['name']}/{k}_architecture.yaml",
              f"{self.model_input}/{vi['name']}/{k}.h5",
              f"{self.model_input}/{vi['name']}/{k}_norm_spline.pkl"
            ]

    # Add best fit if Scan or ScanPoints
    if self.method in ["ScanPointsFromApproximate","ScanPointsFromHessian","Scan","Hessian","HessianParallel","HessianNumerical","DMatrix","ApproximateUncertainty"]:
      inputs += [f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml"]

    # Add hessian parallel
    if self.method == "HessianCollect":
      columns = [i for i in self.Y_columns if self.freeze is None or i not in self.freeze.keys()]
      for column_1_ind, column_1 in enumerate(columns):
        for column_2_ind, column_2 in enumerate(columns):
          if column_1_ind > column_2_ind: continue
          inputs += [f"{self.hessian_input}/hessian{self.extra_file_name}_{column_1}_{column_2}.yaml"]

    # Add hessian 
    if self.method in ["Covariance","CovarianceWithDMatrix","ScanPointsFromHessian"]:
      inputs += [f"{self.hessian_input}/hessian{self.extra_file_name}.yaml"]

    # Add d matrix
    if self.method in ["CovarianceWithDMatrix"]:
      inputs += [f"{self.d_matrix_input}/dmatrix{self.extra_file_name}.yaml"]

    # Add other inputs
    inputs += self.other_input_files

    return inputs


  def _BuildBinYields(self):

    # Define bin_yields dictionary
    bin_yields = {}

    # Loop through files
    for cat, par in self.parameters.items():

      bin_yields[cat] = {}
      for file_name, v in par.items():

        # Open data parameters
        with open(v, 'r') as yaml_file:
          parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Make binned inputs
        rate_param = f"mu_{file_name}" if file_name in self.inference_options["rate_parameters"] else None
        bin_yields[cat][file_name] = GetBinValues(parameters["binned_fit_input"], col=self.binned_fit_morph_col, rate_param=rate_param)

    return bin_yields

    
  def _BuildDataProcessors(self):

    from data_processor import DataProcessor

    # Patch for memory use of gradients
    batch_size = None
    if self.method in ["InitialFit","Scan"] and self.minimisation_method in ["minuit_with_gradients","scipy_with_gradients","custom"]:
      batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_GRADIENTS"))
    elif self.method in ["Hessian","HessianParallel","DMatrix","HessianAndCovariance","HessianDMatrixAndCovariance"]:
      batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_HESSIAN"))
    if self.method == "HessianParallel" and self.hessian_parallel_column_1 in self.non_nn_columns and self.hessian_parallel_column_2 in self.non_nn_columns:
      batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    elif self.method == "HessianParallel" and (self.hessian_parallel_column_2 in self.non_nn_columns or self.hessian_parallel_column_1 in self.non_nn_columns):
      batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_GRADIENTS"))

    # Build dataprocessors
    dps = {}

    for cat, par in self.parameters.items():

      dps[cat] = {}

      if "data" not in self.data_input[cat].keys():

        # Loop through each parameter file
        for k, v in par.items():

          # Open data parameters
          with open(v, 'r') as yaml_file:
            parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

          if self.scale_to_eff_events:
            scale = parameters["eff_events"][self.sim_type][self.val_ind] / parameters["yields"]["nominal"]
          else:
            scale = None

          dps[cat][k] = DataProcessor(
            self.data_input[cat][k],
            "parquet",
            wt_name = "wt",
            batch_size = batch_size,
            options = {
              "parameters" : parameters,
              "scale" : scale,
            }
          )

      else:
          
        dps[cat]["data"] = DataProcessor(
          self.data_input[cat]["data"],
          "parquet",
          batch_size = batch_size,
          options = {}
        )      

    return dps


  def _BuildYieldFunctions(self):

    from yields import Yields

    # Load parameters in
    parameters = {}
    yields = {}
    yields_class = {}

    for cat, par in self.parameters.items():

      parameters[cat] = {}

      for k, v in par.items():
        with open(v, 'r') as yaml_file:
          parameters[cat][k] = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # If scale to effective events, recalculate effective events for combined models
      if self.scale_to_eff_events:
        sum_wt = 0.0
        sum_wt_squared = 0.0
        for k, v in parameters[cat].items():
          sum_wt += v["yields"]["nominal"]
          sum_wt_squared += (v["yields"]["nominal"]**2) / v["eff_events"][self.sim_type][self.val_ind]
        eff_events = (sum_wt**2) / sum_wt_squared
        scale_to = {k: v["yields"]["nominal"]*eff_events/sum_wt for k, v in parameters[cat].items()}
      else:
        scale_to = {k: v["yields"]["nominal"] for k, v in parameters[cat].items()}

      # build yield functions
      yields[cat] = {}
      yields_class[cat] = {}
      for k, v in parameters[cat].items():
        yields_class[cat][k] = Yields(
          scale_to[k],
          lnN = v["yields"]["lnN"],
          physics_model = None,
          rate_param = f"mu_{k}" if k in self.inference_options["rate_parameters"] else None,
        )
        yields[cat][k] = yields_class[cat][k].GetYield

    return yields


  def _BuildDensityModels(self):

    networks = {}
    parameters = {}

    for cat, models in self.density_models.items():

      networks[cat] = {}
      parameters[cat] = {}

      for k, v in models.items():

        # Open parameters
        with open(v["parameters"], 'r') as yaml_file:
          parameters[cat][k] = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Open architecture
        density_model_name = f"{self.model_input}/{v['name']}{self.extra_density_model_name}/{parameters[cat][k]['file_name']}"
        with open(f"{density_model_name}_architecture.yaml", 'r') as yaml_file:
          architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Make model
        networks[cat][k] = InitiateDensityModel(
          architecture,
          v['file_loc'],
          options = {
            "data_parameters" : parameters[cat][k]["density"],
            "file_name" : k,
          }
        )

        networks[cat][k].Load(name=f"{density_model_name}.h5")

    return networks


  def _BuildRegressionModels(self):

    networks = {}
    splines = {}
    parameters = {}

    for cat, models in self.regression_models.items():

      networks[cat] = {}
      splines[cat] = {}
      parameters[cat] = {}

      for k, v in models.items():

        networks[cat][k] = {}
        splines[cat][k] = {}
        parameters[cat][k] = {}

        for vi in v:

          # Open parameters
          with open(vi["parameters"], 'r') as yaml_file:
            parameters[cat][k] = yaml.load(yaml_file, Loader=yaml.FullLoader)

          # Open architecture
          regression_model_name = f"{self.model_input}/{vi['name']}/{parameters[cat][k]['file_name']}"
          with open(f"{regression_model_name}_architecture.yaml", 'r') as yaml_file:
            architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

          # Make model
          networks[cat][k][vi["parameter"]] = InitiateRegressionModel(
            architecture,
            vi['file_loc'],
            options = {
              "data_parameters" : parameters[cat][k]["regression"][vi["parameter"]],
            }
          )

          networks[cat][k][vi["parameter"]].Load(name=f"{regression_model_name}.h5")

          # Make normalising spline
          spline_name = f"{regression_model_name}_norm_spline.pkl"
          if os.path.isfile(spline_name):
            with open(spline_name, 'rb') as f:
              splines[cat][k][vi["parameter"]] = pickle.load(f)

    return networks, splines


  def _BuildClassifierModels(self):

    networks = {}
    splines = {}
    parameters = {}

    for cat, models in self.classifier_models.items():

      networks[cat] = {}
      splines[cat] = {}
      parameters[cat] = {}

      for k, v in models.items():

        networks[cat][k] = {}
        splines[cat][k] = {}
        parameters[cat][k] = {}

        for vi in v:

          # Open parameters
          with open(vi["parameters"], 'r') as yaml_file:
            parameters[cat][k] = yaml.load(yaml_file, Loader=yaml.FullLoader)

          # Open architecture
          classifier_model_name = f"{self.model_input}/{vi['name']}/{parameters[cat][k]['file_name']}"
          with open(f"{classifier_model_name}_architecture.yaml", 'r') as yaml_file:
            architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

          # Make model
          networks[cat][k][vi["parameter"]] = InitiateClassifierModel(
            architecture,
            vi['file_loc'],
            options = {
              "data_parameters" : parameters[cat][k]["classifier"][vi["parameter"]],
            }
          )

          networks[cat][k][vi["parameter"]].Load(name=f"{classifier_model_name}.h5")

          # Make normalising spline
          spline_name = f"{classifier_model_name}_norm_spline.pkl"
          if os.path.isfile(spline_name):
            with open(spline_name, 'rb') as f:
              splines[cat][k][vi["parameter"]] = pickle.load(f)

    return networks, splines


  def _BuildLikelihood(self):

    from likelihood import Likelihood

    if self.verbose:
      print(f"- Building likelihood")
      print(f"- Y_columns={self.Y_columns}")

    #parameters = {}
    #for file_name in self.parameters.keys():
    #  with open(self.parameters[file_name], 'r') as yaml_file:
    #    parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if self.likelihood_type in ["unbinned", "unbinned_extended"]:
      likelihood_inputs = {
        "pdfs" : self._BuildDensityModels(),
        "yields" : self.yields,
      }
      if not self.only_density:
        likelihood_inputs["pdf_shifts_with_regression"], likelihood_inputs["pdf_shifts_with_regression_norm_spline"] = self._BuildRegressionModels()
        likelihood_inputs["pdf_shifts_with_classifier"], likelihood_inputs["pdf_shifts_with_classifier_norm_spline"] = self._BuildClassifierModels()

    elif self.likelihood_type in ["binned", "binned_extended"]:
      likelihood_inputs = {
        "bin_yields" : self._BuildBinYields(),
      }

    if "nuisance_constraints" in self.inference_options.keys():
      constraints = [i for i in self.inference_options["nuisance_constraints"] if i in self.Y_columns]
    else:
      constraints = []

    if self.true_Y is not None and "nuisance_constraints" in self.inference_options.keys():
      constraint_center = self.true_Y.loc[:,constraints]
    else:
      constraint_center = None

    lkld = Likelihood(
      likelihood_inputs, 
      likelihood_type = self.likelihood_type, 
      constraint_center = constraint_center,
      constraints = constraints,
      X_columns = self.X_columns,
      Y_columns = self.Y_columns,
      Y_columns_per_model = self.Y_columns_per_model,
      categories = list(self.parameters.keys())
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