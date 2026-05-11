import copy
import os
import pickle
import yaml

import numpy as np
import pandas as pd

from functools import partial
from pprint import pprint
from scipy.interpolate import CubicSpline, UnivariateSpline

from useful_functions import (
  GetBinValuesParallelised,
  GetDictionaryEntry,
  InitiateDensityModel, 
  InitiateRegressionModel, 
  InitiateClassifierModel,
  MakeDirectories,
  Resample,
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
    self.debug_input = None
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
    self.nn_columns = []
    self.binned_fit_morph_col = None
    self.binned_data_input = None
    self.binned_data_input_parameters_key = None
    self.simplex = {}
    self.bootstrap_ind = None
    self.scan_points_input = {}
    self.remove_lnN_if_rate_param = True
    self.prune_classifier_models = None
    self.classifier_pruning_files = {}
    self.collect_skip_diagonal = False
    self.bootstrap_method = "oversample_to_eff_events" # oversample_to_eff_events, oversample_to_length, undersample_to_eff_events
    self.no_likelihood_print_out = False
    self.integrate_density_with_ratios = False
    self.merge_binned_nuisances = {}
    self.n_integral_events = 10**5
    self.skip_initial_fit = False
    self.binned_from_predicted_bins = False


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

    # Make bin yields or use existing
    self._BuildBinnedFitInput()

    # Make likelihood inputs
    if self.lkld_input is None:
      if self.likelihood_type in ["unbinned", "unbinned_extended","poisson"]:
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
        scipy_method="Nelder-Mead",
      )


    elif self.method == "BootstrapFit":


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

      # Get approximate uncertainties
      self.lkld.GetAndWriteApproximateUncertaintyToYaml(
        self.lkld_input, 
        self.column,
        row=self.true_Y,
        filename=f"{self.data_output}/approximateuncertainty_results_{self.column}{self.extra_file_name}.yaml"
      )

    elif self.method == "UncertaintyFromMinimisation":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Open best fit yaml
      with open(f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
        best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Put best fit in class
      self.lkld.best_fit = np.array(best_fit_info["best_fit"])
      self.lkld.best_fit_nll = best_fit_info["best_fit_nll"] 

      if self.verbose:
        print(f"- Finding uncertainties from minimisation")
        print("- Likelihood output:")

      # Get uncertainties from minimisation
      self.lkld.GetAndWriteUncertaintyFromMinimisationToYaml(
        self.lkld_input, 
        self.column,
        row=self.true_Y,
        filename=f"{self.data_output}/uncertaintyfromminimisation_results_{self.column}{self.extra_file_name}.yaml",
        freeze=self.freeze,
      )

    elif self.method in ["Hessian","HessianParallel","HessianNumerical","HessianNumericalParallel"]:

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
      if self.method in ["HessianParallel", "HessianNumericalParallel"]:
        extra_col_name = f"_{self.hessian_parallel_column_1}_{self.hessian_parallel_column_2}"

      # Get Hessian
      self.lkld.GetAndWriteHessianToYaml(
        self.lkld_input, 
        row=self.true_Y, 
        filename=f"{self.data_output}/hessian{self.extra_file_name}{extra_col_name}.yaml", 
        freeze=self.freeze,
        specific_column_1=self.hessian_parallel_column_1,
        specific_column_2=self.hessian_parallel_column_2,
        numerical = (self.method in ["HessianNumerical", "HessianNumericalParallel"]),
        numerical_method="numdifftools"
      )    

    elif self.method == "HessianCollect":

      first = True
      columns = sorted([i for i in self.Y_columns if self.freeze is None or i not in self.freeze.keys()])

      for column_1_ind, column_1 in enumerate(columns):
        for column_2_ind, column_2 in enumerate(columns):
          if column_1_ind > column_2_ind: continue
          if self.collect_skip_diagonal and column_1 == column_2: continue
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
            for i in range(len(hessian["hessian"])):
              for j in range(len(hessian["hessian"])):
                if hessian["hessian"][i][j] != 0 and total_hessian[i][j] != 0:
                  hessian["hessian"][i][j] = 0
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
        scan_over=hessian["matrix_columns"],
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
        scan_over=hessian["matrix_columns"],
      )

    elif self.method == "ScanPointsFromInput":

      if ":" in self.scan_points_input:
        scan_points_dict = {i.split(':')[0]: i.split(':')[1] for i in self.scan_points_input.split(",")}
        scan_points_for_col = scan_points_dict[self.column]
      else:
        scan_points_for_col = self.scan_points_input

      if "(" in scan_points_for_col:
        numbers = scan_points_for_col.replace("(","").replace(")","").split(",")
      elif "[" in scan_points_for_col:
        numbers = scan_points_for_col.replace("[","").replace("]","").split(",")

      points = [float(i) for i in np.linspace(float(numbers[0]), float(numbers[1]), int(self.number_of_scan_points))]


      # Make scan ranges
      self.lkld.GetAndWriteScanRangesToYaml(
        self.lkld_input, 
        self.column,
        row=self.true_Y,
        filename=f"{self.data_output}/scan_ranges_{self.column}{self.extra_file_name}.yaml",
        scan_values=points
      )


    elif self.method == "Scan":

      if self.verbose:
        print(f"- Loading best fit into likelihood")

      # Put best fit in class
      if not self.skip_initial_fit:      
        with open(f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml", 'r') as yaml_file:
          best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.lkld.best_fit = np.array(best_fit_info["best_fit"])
        self.lkld.best_fit_nll = best_fit_info["best_fit_nll"] 
      else:
        self.lkld.best_fit = np.array([self.initial_best_fit_guess.loc[0, col] for col in self.lkld.Y_columns])
        self.lkld.best_fit_nll = 0.0

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

      #for j in self.other_input.split(":"):
      #  self.lkld.Run(self.lkld_input, [float(i) for i in j.split(',')])
      for j in self.debug_input:
        guess = self.initial_best_fit_guess.copy()
        for key, val in j.items():
          guess.loc[0, key] = float(val)
        self.lkld.Run(self.lkld_input, [guess.loc[0,v] for v in self.lkld.Y_columns])


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
    if self.method in ["HessianParallel", "HessianNumericalParallel"]:
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
    if self.method in ["ScanPointsFromApproximate","ScanPointsFromHessian","ScanPointsFromInput"]:
      outputs += [f"{self.data_output}/scan_ranges_{self.column}{self.extra_file_name}.yaml",]

    # Add scan values
    if self.method == "Scan":
      outputs += [f"{self.data_output}/scan_values_{self.column}{self.extra_file_name}_{self.scan_ind}.yaml",]

    # Add approximate uncertainty
    if self.method == "ApproximateUncertainty":
      outputs += [f"{self.data_output}/approximateuncertainty_results_{self.column}{self.extra_file_name}.yaml"]

    # Add uncertainty from minimisation
    if self.method == "UncertaintyFromMinimisation":
      outputs += [f"{self.data_output}/uncertaintyfromminimisation_results_{self.column}{self.extra_file_name}.yaml"]

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
    if self.likelihood_type in ["unbinned", "unbinned_extended","poisson"]:
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
      if not self.only_density:
        for cat, models in self.regression_models.items():
          for k, v in models.items():
            for vi in v:
              inputs += [
                f"{self.model_input}/{vi['name']}/{k}_architecture.yaml",
                f"{self.model_input}/{vi['name']}/{k}.h5",
              ]
              if not self.integrate_density_with_ratios:
                inputs += [f"{self.model_input}/{vi['name']}/{k}_norm_spline.pkl"]



        # Add classifier model inputs
        for cat, models in self.classifier_models.items():
          for k, v in models.items():
            for vi in v:
              inputs += [
                f"{self.model_input}/{vi['name']}/{k}_architecture.yaml",
                f"{self.model_input}/{vi['name']}/{k}.h5",
              ]
              if not self.integrate_density_with_ratios:
                inputs += [f"{self.model_input}/{vi['name']}/{k}_norm_spline.pkl"]
              if self.prune_classifier_models is not None:
                if k in self.prune_classifier_models.keys():
                  inputs += [self.classifier_pruning_files[cat][k][vi['parameter']]]
          
    # Add best fit if Scan or ScanPoints
    if self.method in ["ScanPointsFromApproximate","ScanPointsFromHessian","Scan","Hessian","HessianParallel","HessianNumerical","HessianNumericalParallel","DMatrix","ApproximateUncertainty","UncertaintyFromMinimisation"]:
      if not (self.method == "Scan" and self.skip_initial_fit): 
        inputs += [f"{self.best_fit_input}/best_fit{self.extra_file_name}.yaml"]

    # Add hessian parallel
    if self.method == "HessianCollect":
      columns = [i for i in self.Y_columns if self.freeze is None or i not in self.freeze.keys()]
      for column_1_ind, column_1 in enumerate(columns):
        for column_2_ind, column_2 in enumerate(columns):
          if column_1_ind > column_2_ind: continue
          if self.collect_skip_diagonal and column_1 == column_2: continue
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
    first_loop = True
    shape_direction = {}

    # Open all parameters
    all_parameters = {}
    for cat, par in self.parameters.items():
      all_parameters[cat] = {}
      for file_name, v in par.items():
        with open(v, 'r') as yaml_file:
          parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
        all_parameters[cat][file_name] = copy.deepcopy(parameters)


    # Remove total rate of nuisances if remove_lnN_if_rate_param is True and this file is a rate parameter, as the total rate will be floating and the lnN will not make sense
    all_nuisances = []
    for cat, par in all_parameters.items():
      for file_name, v in par.items():
        if (self.remove_lnN_if_rate_param and file_name in self.inference_options["rate_parameters"]): 
          for bin in all_parameters[cat][file_name]["binned_fit_input"]:
            for poi_val, poi_val_info in bin["yields"].items():
              all_nuisances += list(poi_val_info["lnN"].keys())
    all_nuisances = list(set(all_nuisances))

    total_count = {}
    nominal_counts = {}
    for cat, par in all_parameters.items():
      for file_name, v in par.items():
        if not (self.remove_lnN_if_rate_param and file_name in self.inference_options["rate_parameters"]): continue
        if file_name not in total_count.keys():
          total_count[file_name] = {}
        if file_name not in nominal_counts.keys():
          nominal_counts[file_name] = {}
        parameters = copy.deepcopy(all_parameters[cat][file_name])
        for bin_ind, bin in enumerate(parameters["binned_fit_input"]):
          for poi_val, poi_val_info in bin["yields"].items():
            if poi_val not in total_count[file_name].keys():
              total_count[file_name][poi_val] = {}
            if poi_val not in nominal_counts[file_name].keys():
              nominal_counts[file_name][poi_val] = 0.0
            nominal_counts[file_name][poi_val] += bin["yields"][poi_val]["nominal"]
            for nuisance in all_nuisances:
              if nuisance not in total_count[file_name][poi_val].keys():
                total_count[file_name][poi_val][nuisance] = {"up": 0.0, "down": 0.0}
              if nuisance not in poi_val_info["lnN"].keys(): 
                total_count[file_name][poi_val][nuisance]["up"] += bin["yields"][poi_val]["nominal"]
                total_count[file_name][poi_val][nuisance]["down"] += bin["yields"][poi_val]["nominal"]
              else:
                lnN = poi_val_info["lnN"][nuisance]
                total_count[file_name][poi_val][nuisance]["up"] += bin["yields"][poi_val]["nominal"] * lnN[1]
                total_count[file_name][poi_val][nuisance]["down"] += bin["yields"][poi_val]["nominal"] * lnN[0]

    for cat, par in all_parameters.items():
      for file_name, poi_val_info in total_count.items():
        for poi_val, nuisance_info in poi_val_info.items():
          if nominal_counts[file_name][poi_val] == 0: continue
          for nuisance, counts in nuisance_info.items():
            up_norm = counts["up"] / nominal_counts[file_name][poi_val]
            down_norm = counts["down"] / nominal_counts[file_name][poi_val]
            for bin_ind, bin in enumerate(all_parameters[cat][file_name]["binned_fit_input"]):
              if poi_val not in bin["yields"].keys(): continue
              if nuisance not in bin["yields"][poi_val]["lnN"].keys(): continue
              new_up = all_parameters[cat][file_name]["binned_fit_input"][bin_ind]["yields"][poi_val]["lnN"][nuisance][1] / up_norm
              new_down = all_parameters[cat][file_name]["binned_fit_input"][bin_ind]["yields"][poi_val]["lnN"][nuisance][0] / down_norm
              all_parameters[cat][file_name]["binned_fit_input"][bin_ind]["yields"][poi_val]["lnN"][nuisance] = [new_down, new_up]

    # Do final loop
    for cat, par in all_parameters.items():

      bin_yields[cat] = {}
      for file_name, v in par.items():

        # Open data parameters
        parameters = copy.deepcopy(all_parameters[cat][file_name])

        # Do nuisance merging
        if len(self.merge_binned_nuisances) > 0:

          if first_loop:

            first_loop = False

            # Change Y_colums and nuisance_constraints and initial_best_fit_guess to merged nuisances and self.true_Y
            for merged_nuisance, nuisances_to_merge in self.merge_binned_nuisances.items():
              self.Y_columns += [merged_nuisance]
              self.inference_options["nuisance_constraints"] += [merged_nuisance]
              self.initial_best_fit_guess.loc[0, merged_nuisance] = 0.0
              self.true_Y.loc[0, merged_nuisance] = 0.0
              for nuisance_to_merge in nuisances_to_merge:
                if nuisance_to_merge in self.Y_columns:
                  self.Y_columns.remove(nuisance_to_merge)
                if nuisance_to_merge in self.inference_options["nuisance_constraints"]:
                  self.inference_options["nuisance_constraints"].remove(nuisance_to_merge)
                if nuisance_to_merge in self.initial_best_fit_guess.columns:
                  self.initial_best_fit_guess.drop(columns=[nuisance_to_merge], inplace=True)
                if nuisance_to_merge in self.true_Y.columns:
                  self.true_Y.drop(columns=[nuisance_to_merge], inplace=True)

            # Get nominal poi_val
            poi_val = self.initial_best_fit_guess.loc[0, self.binned_fit_morph_col] if self.binned_fit_morph_col in self.Y_columns_per_model[file_name] else "all"
            
          # Need to get direction of shape
          for _, merged_nuisances in self.merge_binned_nuisances.items():
            for merged_nuisance in merged_nuisances:
              if merged_nuisance in shape_direction.keys(): continue
              n_nom = 0
              n_invert = 0
              for bin in parameters["binned_fit_input"]:
                if merged_nuisance not in bin["yields"][poi_val]["lnN"].keys():
                  n_nom += 1
                else:
                  if bin["yields"][poi_val]["lnN"][merged_nuisance][1] > bin["yields"][poi_val]["lnN"][merged_nuisance][0]:
                    n_nom += 1
                  else:
                    n_invert += 1
              if n_nom > n_invert:
                shape_direction[merged_nuisance] = 1
              else:
                shape_direction[merged_nuisance] = 0

          # Merge nuisances
          for bin_ind, bin in enumerate(copy.deepcopy(parameters["binned_fit_input"])):
            for poi_val, poi_val_info in bin["yields"].items():
              new_lnN = copy.deepcopy(poi_val_info["lnN"])
              for merged_nuisance, nuisances_to_merge in self.merge_binned_nuisances.items():
                new_lnN[merged_nuisance] = [1.0, 1.0]
                for nuisance_to_merge in nuisances_to_merge:

                  if nuisance_to_merge not in poi_val_info["lnN"].keys(): continue

                  # Need to decide which direction is up and which is down
                  if shape_direction[nuisance_to_merge] == 1:
                    nominal_nui_direction = poi_val_info["lnN"][nuisance_to_merge]
                  else:
                    nominal_nui_direction = [poi_val_info["lnN"][nuisance_to_merge][1], poi_val_info["lnN"][nuisance_to_merge][0]]
                  sum_nominal_nuis = [nominal_nui_direction[0]+ new_lnN[merged_nuisance][0], nominal_nui_direction[1]+new_lnN[merged_nuisance][1]]
                  if sum_nominal_nuis[1] > sum_nominal_nuis[0]:
                    bin_direction = 1
                  else:
                    bin_direction = -1

                  # Combine in quadrature
                  if shape_direction[nuisance_to_merge] == 1:
                    new_lnN[merged_nuisance][0] = 1 - bin_direction * (((new_lnN[merged_nuisance][0]-1)**2 + (poi_val_info["lnN"][nuisance_to_merge][0]-1)**2)**0.5)
                    new_lnN[merged_nuisance][1] = 1 + bin_direction * (((new_lnN[merged_nuisance][1]-1)**2 + (poi_val_info["lnN"][nuisance_to_merge][1]-1)**2)**0.5)
                  else:
                    new_lnN[merged_nuisance][0] = 1 - bin_direction * (((new_lnN[merged_nuisance][0]-1)**2 + (poi_val_info["lnN"][nuisance_to_merge][0]-1)**2)**0.5)
                    new_lnN[merged_nuisance][1] = 1 + bin_direction * (((new_lnN[merged_nuisance][1]-1)**2 + (poi_val_info["lnN"][nuisance_to_merge][1]-1)**2)**0.5)

                  # Remove nuisance_to_merge from new_lnN
                  del new_lnN[nuisance_to_merge]

                # Update new_lnN for merged_nuisance
                parameters["binned_fit_input"][bin_ind]["yields"][poi_val]["lnN"] = new_lnN


        # Make binned inputs
        rate_param = f"mu_{file_name}" if file_name in self.inference_options["rate_parameters"] else None

        #bin_yields[cat][file_name] = GetBinValues(parameters["binned_fit_input"], col=self.binned_fit_morph_col, rate_param=rate_param)
        bin_yields[cat][file_name] = GetBinValuesParallelised(parameters["binned_fit_input"], col=self.binned_fit_morph_col, rate_param=rate_param)

    return bin_yields

    
  def _BuildDataProcessors(self):

    from data_processor import DataProcessor

    # Patch for memory use of gradients
    batch_size = None
    if self.method in ["InitialFit","Scan","BootstrapFit"] and self.minimisation_method in ["minuit-with-gradients","scipy-with-gradients","custom"]:
      batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_GRADIENTS"))
    elif self.method in ["Hessian","DMatrix"]:
      batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_HESSIAN"))
    elif self.method == "HessianParallel" and self.hessian_parallel_column_1 in self.non_nn_columns and self.hessian_parallel_column_2 in self.non_nn_columns:
      batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    elif self.method == "HessianParallel" and (self.hessian_parallel_column_2 in self.non_nn_columns or self.hessian_parallel_column_1 in self.non_nn_columns):
      batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_GRADIENTS"))
    elif self.method == "HessianParallel":
      batch_size = int(os.getenv("EVENTS_PER_BATCH_FOR_HESSIAN"))
    else:
      batch_size = int(os.getenv("EVENTS_PER_BATCH"))

    functions_to_apply = []
    if self.bootstrap_ind is not None:
      def bootstrap(df):
        columns = [i for i in df.columns.tolist() if i != "wt"]
        X, wt = df.drop(columns=["wt"]), df["wt"] 
        if self.bootstrap_method == "oversample_to_length":
          X, wt = Resample(X.to_numpy(), wt.to_numpy().flatten(), method="oversample", sample_size="length", keep_weights=True, total_scale="eff_events", seed=self.bootstrap_ind)
        elif self.bootstrap_method == "oversample_to_eff_events":
          X, wt = Resample(X.to_numpy(), wt.to_numpy().flatten(), method="oversample", sample_size="eff_events", keep_weights=False, total_scale="eff_events", seed=self.bootstrap_ind)
        elif self.bootstrap_method == "undersample_to_eff_events":
          X, wt = Resample(X.to_numpy(), wt.to_numpy().flatten(), method="undersample", sample_size="eff_events", keep_weights=False, total_scale="eff_events", seed=self.bootstrap_ind)
        else:
          raise ValueError(f"Invalid bootstrap method: {self.bootstrap_method}")
        df = pd.DataFrame(X, columns=columns)
        df.loc[:,"wt"] = wt
        return df
      functions_to_apply = [bootstrap]
    

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
              "functions" : functions_to_apply,
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

    #from yields_parallelised import Yields
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

        lnN_yields = v["yields"]["lnN"]
        if self.remove_lnN_if_rate_param and k in self.inference_options["rate_parameters"]:
          lnN_yields = {}

        yields_class[cat][k] = Yields(
          scale_to[k],
          lnN = lnN_yields,
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

        # Add neural network parameters
        self.nn_columns += parameters[cat][k]["density"]["Y_columns"]
        self.nn_columns = list(set(self.nn_columns))

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

          # Add neural network parameters
          self.nn_columns += [vi["parameter"]]
          self.nn_columns = list(set(self.nn_columns))

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

          # Check if we need to prune this model
          prune = False
          if self.prune_classifier_models is not None:
            with open(self.classifier_pruning_files[cat][k][vi['parameter']], 'r') as yaml_file:
              pruning_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
            prune = True
            for pruning_key, pruning_val in self.prune_classifier_models.items():
              if pruning_info[pruning_key] > pruning_val:
                prune = False
          if prune: 
            if self.verbose:
              print(f"- Pruning classifier model for category {cat}, model {k}, parameter {vi['parameter']}")
            continue

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

          # Add neural network parameters
          self.nn_columns += [vi["parameter"]]
          self.nn_columns = list(set(self.nn_columns))

    if self.verbose:
      print(f"- Columns using neural networks: {self.nn_columns}")

    # Cap spline ranges to avoid issues in likelihood
    def capped_spline(x, spline, x_min, x_max):
      x = np.asarray(x)
      y = np.empty_like(x, dtype=float)
      mask = (x >= x_min) & (x <= x_max)
      y[mask] = spline(x[mask])
      y[~mask] = np.nan
      return y

    capped_splines = {}
    for cat in splines.keys():
      capped_splines[cat] = {}
      for k in splines[cat].keys():
        capped_splines[cat][k] = {}
        for vi in splines[cat][k].keys():
          if isinstance(splines[cat][k][vi], CubicSpline):
            capped_splines[cat][k][vi] = partial(capped_spline, spline=splines[cat][k][vi], x_min=splines[cat][k][vi].x[0], x_max=splines[cat][k][vi].x[-1])
          elif isinstance(splines[cat][k][vi], UnivariateSpline):
            capped_splines[cat][k][vi] = partial(capped_spline, spline=splines[cat][k][vi], x_min=splines[cat][k][vi].get_knots()[0], x_max=splines[cat][k][vi].get_knots()[-1])
          else:
            raise ValueError(f"Invalid spline type: {type(splines[cat][k][vi])}")

    return networks, capped_splines


  def _BuildLikelihood(self):

    from likelihood import Likelihood

    if self.verbose:
      print(f"- Building likelihood")
      print(f"- Y_columns={self.Y_columns}")

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
    elif self.likelihood_type in ["poisson"]:
      likelihood_inputs = {
        "yields" : self.yields,
      }

    if "nuisance_constraints" in self.inference_options.keys():
      constraints = [i for i in self.inference_options["nuisance_constraints"] if i in self.Y_columns]
    else:
      constraints = []

    if self.true_Y is not None and self.verbose:
      print(f"- Using truth dataset for likelihood evaluation:")
      print(self.true_Y)

    if self.true_Y is not None and "nuisance_constraints" in self.inference_options.keys():
      constraint_center = self.true_Y.loc[:,constraints]
      if constraint_center.shape[0] > 1:
        if self.verbose:
          print(f"- Using truth for constraint center:")
          print(constraint_center)
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
      categories = list(self.parameters.keys()),
      simplex = self.simplex,
      nn_columns = self.nn_columns,
    )

    lkld.integrate_density_with_ratios = self.integrate_density_with_ratios
    lkld.no_print_minimisation_step = self.no_likelihood_print_out
    lkld.n_integral_events = self.n_integral_events
    lkld.integral_events_per_batch = int(os.getenv("EVENTS_PER_BATCH"))

    return lkld
  
  def _BuildBinnedFitInput(self):

    if not self.binned_from_predicted_bins:

      if (self.binned_data_input is None or self.binned_data_input == {}) and self.binned_data_input_parameters_key is not None and self.likelihood_type in ["binned","binned_extended"]:
        self.binned_data_input = {}
        for cat, pars in self.parameters.items():
          first = True
          for k, v in pars.items():
            with open(v, 'r') as yaml_file:
              parameters = yaml.load(yaml_file, Loader=yaml.CLoader)
            entry = GetDictionaryEntry(parameters, self.binned_data_input_parameters_key[cat][k])
            if first:
              self.binned_data_input[cat] = np.array(entry)
              first = False
            else:
              self.binned_data_input[cat] += np.array(entry)

    else:
      
      yields = self._BuildBinYields()
      self.binned_data_input = {}
      for cat in yields.keys():
        for name, func in yields[cat].items():
          if cat not in self.binned_data_input.keys():
            self.binned_data_input[cat] = np.array(func(self.true_Y))
          else:
            self.binned_data_input[cat] += np.array(func(self.true_Y))