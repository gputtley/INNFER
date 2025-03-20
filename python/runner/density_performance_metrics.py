import copy
import yaml

import numpy as np
import pandas as pd

from functools import partial

from histogram_metrics import HistogramMetrics
from multidim_metrics import MultiDimMetrics
from useful_functions import MakeDirectories, GetYName, SplitValidationParameters

class DensityPerformanceMetrics():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.model = None
    self.parameters = None
    self.architecture = None
    self.val_loop = []
    self.pois = None
    self.nuisances = None
    self.cfg_name = None
  
    self.data_output = "data/"
    self.verbose = True
    
    self.do_latex_table = True

    self.do_loss = True
    self.loss_datasets = ["test","test_inf","val"]

    self.do_histogram_metrics = True
    self.do_chi_squared = True
    self.do_kl_divergence = False
    self.histogram_datasets = ["test","test_inf","val"]

    self.do_multidimensional_dataset_metrics = True
    self.do_bdt_separation = True
    self.do_wasserstein = True
    self.do_sliced_wasserstein = True
    self.multidimensional_datasets = ["test","test_inf","val"]

    self.do_inference = True
    self.inference_datasets = ["test_inf","val"]

    self.test_name = "test"
    self.save_extra_name = ""
    self.split_validation_files = False


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
    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      self.open_parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the architecture in
    if self.verbose:
      print("- Loading in the architecture")
    with open(self.architecture, 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build model
    if self.verbose:
      print("- Building the model")
    from network import Network
    self.network = Network(
      f"{self.open_parameters['file_loc']}/X_train.parquet",
      f"{self.open_parameters['file_loc']}/Y_train.parquet", 
      f"{self.open_parameters['file_loc']}/wt_train.parquet", 
      f"{self.open_parameters['file_loc']}/X_{self.test_name}.parquet",
      f"{self.open_parameters['file_loc']}/Y_{self.test_name}.parquet", 
      f"{self.open_parameters['file_loc']}/wt_{self.test_name}.parquet",
      options = {
        **architecture,
        **{
          "data_parameters" : self.open_parameters
        }
      }
    )  
    
    # Loading model
    if self.verbose:
      print("- Loading the model")
    self.network.Load(name=self.model)

    # Set up metrics dictionary
    self.metrics = {} 

    # Get the loss values
    if self.do_loss:
      if self.verbose:
        print("- Getting the losses")
      self.DoLoss()

    # Get histogram metrics
    if self.do_histogram_metrics:
      if self.verbose:
        print("- Getting histogram metrics")
      self.DoHistogramMetrics()

    # Get multidimensional dataset metrics
    if self.do_multidimensional_dataset_metrics:
      if self.verbose:
        print("- Getting multidimensional dataset metrics")
      self.DoMultiDimensionalDatasetMetrics()

    if self.do_inference and len(self.open_parameters["Y_columns"]) > 0:
      if self.verbose:
        print("- Getting chi squared from quick inference")
      self.DoInference()

    # Write to yaml
    if self.verbose:
      print("- Writing metrics yaml")
    output_name = f"{self.data_output}/metrics{self.save_extra_name}.yaml"
    MakeDirectories(output_name)
    with open(output_name, 'w') as yaml_file:
      yaml.dump(self.metrics, yaml_file, default_flow_style=False) 

    # Print metrics
    for metric in sorted(list(self.metrics.keys())):
      if not isinstance(self.metrics[metric], dict):
        print(f"{metric} : {self.metrics[metric]}")
      else:
        print(f"{metric} :")
        for k1 in sorted(list(self.metrics[metric].keys())):
          if not isinstance(self.metrics[metric][k1], dict):
            print(f"  {k1} : {self.metrics[metric][k1]}")
          else:
            print(f"  {k1} :")
            for k2 in sorted(list(self.metrics[metric][k1].keys())):
              print(f"    {k2} : {self.metrics[metric][k1][k2]}")          

  def DoLoss(self):

    # Get loss values
    for data_type in self.loss_datasets:
      if self.verbose:
        print(f"  - Getting loss for {data_type}")
      self.metrics[f"loss_{data_type}"] = self.network.Loss(
        f"{self.open_parameters['file_loc']}/X_{data_type}.parquet", 
        f"{self.open_parameters['file_loc']}/Y_{data_type}.parquet", 
        f"{self.open_parameters['file_loc']}/wt_{data_type}.parquet"
      )

  def DoHistogramMetrics(self):

    # Initialise histogram metrics
    hm = HistogramMetrics(
      self.network,
      self.open_parameters,
      {data_type:[f"{self.open_parameters['file_loc']}/X_{data_type}.parquet", f"{self.open_parameters['file_loc']}/Y_{data_type}.parquet", f"{self.open_parameters['file_loc']}/wt_{data_type}.parquet"] for data_type in self.histogram_datasets}
    )

    # Get chi squared values
    if self.do_chi_squared:
      if self.verbose:
        print(" - Doing chi squared")
      chi_squared, dof_for_chi_squared, chi_squared_per_dof = hm.GetChiSquared()
      self.metrics = {**self.metrics, **chi_squared, **dof_for_chi_squared, **chi_squared_per_dof}

    # Get KL divergence values
    if self.do_kl_divergence:
      if self.verbose:
        print(" - Doing KL divergence")
      kl_divergence = hm.GetKLDivergence()
      self.metrics = {**self.metrics, **kl_divergence}


  def DoMultiDimensionalDatasetMetrics(self):

    # Initialise multi metrics
    mm = MultiDimMetrics(
      self.network,
      self.open_parameters,
      {data_type:[f"{self.open_parameters['file_loc']}/X_{data_type}.parquet", f"{self.open_parameters['file_loc']}/Y_{data_type}.parquet", f"{self.open_parameters['file_loc']}/wt_{data_type}.parquet"] for data_type in self.multidimensional_datasets}
    )
    mm.verbose = self.verbose

    # Get BDT separation metric
    if self.do_bdt_separation:
      if self.verbose:
        print(" - Adding BDT separation")
      mm.AddBDTSeparation()
    
    # Get Wasserstein metric
    if self.do_wasserstein:
      if self.verbose:
        print(" - Adding Wasserstein")
      mm.AddWassersteinUnbinned()

    # Get sliced Wasserstein metric
    if self.do_sliced_wasserstein:
      if self.verbose:
        print(" - Adding sliced Wasserstein")
      mm.AddWassersteinSliced()

    # Run metrics
    multidim_metrics = mm.Run()
    self.metrics = {**self.metrics, **multidim_metrics}


  def DoInference(self):

    for inf_test_name in self.inference_datasets:
    
      # Build yields
      from yields import Yields
      eff_events_class = Yields(
        pd.read_parquet(self.open_parameters['yield_loc']), 
        self.pois, 
        self.nuisances, 
        self.open_parameters["file_name"],
        method="default", 
        column_name=f"effective_events_{inf_test_name}"
      )

      # Build likelihood
      from likelihood import Likelihood
      lkld = Likelihood(
        {
          "pdfs" : {self.open_parameters["file_name"] : self.network},
        },
        likelihood_type = "unbinned", 
        data_parameters = {self.open_parameters["file_name"] : self.open_parameters},
      )

      # Loop through validation values
      orig_parameters = copy.deepcopy(self.open_parameters)
      inf_chi_squared = {}
      inf_dist = {}
      for loop_ind, loop in enumerate(self.val_loop):

        if self.split_validation_files:
          cfg = {"files" : {self.open_parameters["file_name"] : None}, "name" : self.cfg_name}
          parameters_file_name = SplitValidationParameters(loop, self.open_parameters["file_name"], loop_ind, cfg)
          with open(parameters_file_name, 'r') as yaml_file:
            self.open_parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

        if self.verbose:
          print(f"  - Running unbinned likelihood fit for the {inf_test_name} dataset and Y:")
          print(loop["row"])

        # Build test data loader
        from data_processor import DataProcessor
        dps = DataProcessor(
          [[f"{self.open_parameters['file_loc']}/X_{inf_test_name}.parquet", f"{self.open_parameters['file_loc']}/Y_{inf_test_name}.parquet", f"{self.open_parameters['file_loc']}/wt_{inf_test_name}.parquet"]],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : self.open_parameters,
            "selection" : " & ".join([f"({col}=={loop['row'].loc[:,col].iloc[0]})" for col in loop['row'].columns]),
            "scale" : eff_events_class.GetYield(loop["row"]),
            "functions" : ["untransform"]
          }
        )

        # Skip if empty
        if dps.GetFull(method="count") == 0: continue

        # Do initial fit
        lkld.GetBestFit([dps], loop["initial_best_fit_guess"])

        # Get uncertainty
        y_name = GetYName(loop['row'], purpose="file")
        inf_chi_squared[y_name] = {}
        inf_dist[y_name] = {}
        for col in loop['row'].columns:
          if self.verbose:
            print(f"  - Finding uncertainty estimates for {col}")
          uncert = lkld.GetApproximateUncertainty([dps], col)
          col_index = list(loop['row'].columns).index(col)
          true_value = float(loop['row'].loc[0,col])
          if true_value > lkld.best_fit[col_index]:
            inf_chi_squared[y_name][col] = float(((true_value - lkld.best_fit[col_index])**2) / (uncert[1]**2))
          else:
            inf_chi_squared[y_name][col] = float(((true_value - lkld.best_fit[col_index])**2) / (uncert[-1]**2))
          inf_dist[y_name][col] = abs(float(true_value - lkld.best_fit[col_index]))

      # Reset parameters
      self.open_parameters = copy.deepcopy(orig_parameters)

      # Get chi squared values
      total_sum = 0.0
      total_count = 0.0
      for val_name, val_dict in inf_chi_squared.items():
        total_sum += np.sum(list(val_dict.values()))
        total_count += len(list(val_dict.values()))
        inf_chi_squared[val_name]["all"] = float(np.sum(list(val_dict.values())))/len(list(val_dict.values()))
      inf_chi_squared["all"] = float(total_sum/total_count)
      self.metrics[f"inference_chi_squared_{inf_test_name}"] = inf_chi_squared

      # Get distance values
      total_sum = 0.0
      total_count = 0.0
      for val_name, val_dict in inf_dist.items():
        total_sum += np.sum(list(val_dict.values()))
        total_count += len(list(val_dict.values()))
        inf_dist[val_name]["all"] = float(np.sum(list(val_dict.values())))/len(list(val_dict.values()))
      inf_dist["all"] = float(total_sum/total_count)
      self.metrics[f"inference_distance_{inf_test_name}"] = inf_dist


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

        