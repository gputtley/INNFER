import os
import yaml

import numpy as np
import pandas as pd

from functools import partial

from data_processor import DataProcessor
from histogram_metrics import HistogramMetrics
from make_asimov import MakeAsimov
from multidim_metrics import MultiDimMetrics
from yields import Yields

from useful_functions import (
  GetDefaultsInModel,
  GetModelLoop,
  GetValidationLoop,
  InitiateDensityModel, 
  LoadConfig, 
  MakeDirectories,
  SkipEmptyDataset,
  SkipNonDensity,
)

class DensityPerformanceMetrics():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.cfg = None
    self.parameters = None
  
    self.category = None
    self.file_name = None
    self.model_input = "models/"
    self.extra_model_dir = ""
    self.data_output = "data/"
    self.verbose = True
    self.n_asimov_events = 10**6
    self.asimov_seed = 0
    self.file_loc = None
    self.val_file_loc = None
    
    self.do_loss = True
    self.loss_datasets = ["train","test"]

    self.do_histogram_metrics = True
    self.do_chi_squared = True
    self.do_kl_divergence = False
    self.histogram_datasets = ["test_inf","val"]

    self.do_multidimensional_dataset_metrics = True
    self.do_bdt_separation = True
    self.do_wasserstein = True
    self.do_sliced_wasserstein = True
    self.do_kmeans_chi_squared = False
    self.multidimensional_datasets = ["test_inf","val"]

    self.do_inference = True
    self.inference_datasets = ["test_inf","val"]

    self.save_extra_name = ""
    self.metrics_save_extra_name = ""
    self.tidy_up_asimov = False
    self.asimov_input = None
    self.synth_vs_synth = False
    self.alternative_asimov_seed_shift = 0
    self.alternative_asimov_seed = 1
    self.use_eff_events = False


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
    # Load config
    if self.verbose:
      print("- Loading in the config")
    self.open_cfg = LoadConfig(self.cfg)

    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      self.open_parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if self.do_inference or self.do_loss:

      density_model_name = f"{self.model_input}/{self.extra_model_dir}/{self.file_name}{self.save_extra_name}"

      # Load the architecture in
      if self.verbose:
        print("- Loading in the architecture")
      with open(f"{density_model_name}_architecture.yaml", 'r') as yaml_file:
        architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
      
      # Build model
      if self.verbose:
        print("- Building the model")
      self.network = InitiateDensityModel(
        architecture,
        self.open_parameters['density']['file_loc'],
        test_name = "test",
        options = {
          "data_parameters" : self.open_parameters["density"],
          "file_name" : self.file_name,
        }
      )

      # Load weights into model
      if self.verbose:
        print(f"- Loading the density model {density_model_name}")
      self.network.Load(name=f"{density_model_name}.h5")

    # Set up metrics dictionary
    self.metrics = {} 

    # Get the loss values
    if self.do_loss:
      if self.verbose:
        print("- Getting the losses")
      self.DoLoss()

    # Make asimov datasets
    if self.do_histogram_metrics or self.do_multidimensional_dataset_metrics:

      if self.verbose:
        print("- Making asimov datasets")

      for data_type in sorted(list(set(self.histogram_datasets+self.multidimensional_datasets+self.inference_datasets))):

        ma = MakeAsimov()
        for val_ind, val_info in enumerate(GetValidationLoop(self.open_cfg, self.file_name)):
          if SkipNonDensity(self.open_cfg, self.file_name, val_info, skip_non_density=True): continue
          if SkipEmptyDataset(self.open_cfg, self.file_name, data_type, val_info): continue


          if self.use_eff_events:
            n_events = int(np.ceil(self.open_parameters["eff_events"][data_type][val_ind]))
          else:
            n_events = self.n_asimov_events
 
          if n_events == 0: continue

          if self.asimov_input is None:

            ma = MakeAsimov()
            ma.Configure({
                "cfg" : self.cfg,
                "density_model" : GetModelLoop(self.open_cfg, model_file_name=self.file_name, only_density=True, specific_category=self.category)[0],
                "model_input" : self.model_input,
                "model_extra_name" : self.save_extra_name,
                "parameters" : self.parameters,
                "data_output" : f"{self.data_output}/val_ind_{val_ind}{self.metrics_save_extra_name}_seed_{self.asimov_seed}_for_{data_type}",
                "n_asimov_events" : n_events,
                "seed" : self.asimov_seed,
                "val_info" : val_info,
                "val_ind" : val_ind,
                "only_density" : True,
                "add_truth" : True,
                "verbose" : False,
              }
            )
            ma.Run()

          if self.synth_vs_synth:

            if self.alternative_asimov_seed_shift < 0: 
              self.alternative_asimov_seed = 1
            else:
              self.alternative_asimov_seed = self.asimov_seed + self.alternative_asimov_seed_shift + 1

              ma2 = MakeAsimov()
              ma2.Configure({
                  "cfg" : self.cfg,
                  "density_model" : GetModelLoop(self.open_cfg, model_file_name=self.file_name, only_density=True)[0],
                  "model_input" : self.model_input,
                  "model_extra_name" : self.save_extra_name,
                  "parameters" : self.parameters,
                  "data_output" : f"{self.data_output}/val_ind_{val_ind}{self.metrics_save_extra_name}_seed_{self.alternative_asimov_seed}_for_{data_type}",
                  "n_asimov_events" : n_events,
                  "seed" : self.alternative_asimov_seed,
                  "val_info" : val_info,
                  "val_ind" : val_ind,
                  "only_density" : True,
                  "add_truth" : True,
                  "verbose" : False,
                }
              )
              ma2.Run()        

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

    # Get inference metrics
    if self.do_inference and len(self.open_parameters["density"]["Y_columns"]) > 0:
      if self.verbose:
        print("- Getting chi squared from quick inference")
      self.DoInference()

    # Write to yaml
    if self.verbose:
      print("- Writing metrics yaml")
    output_name = f"{self.data_output}/metrics{self.save_extra_name}{self.metrics_save_extra_name}.yaml"
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

    # Tidy up asimov
    if self.verbose:
      print("- Tidying up asimov datasets")
    for data_type in sorted(list(set(self.histogram_datasets+self.multidimensional_datasets+self.inference_datasets))):
      for val_ind, val_info in enumerate(GetValidationLoop(self.open_cfg, self.file_name)):
        if SkipNonDensity(self.open_cfg, self.file_name, val_info, skip_non_density=True): continue
        if SkipEmptyDataset(self.open_cfg, self.file_name, data_type, val_info): continue
        if self.tidy_up_asimov:
          os.system(f"rm -rf {self.data_output}/val_ind_{val_ind}{self.metrics_save_extra_name}_seed_{self.asimov_seed}_for_{data_type}")
        if self.synth_vs_synth:
          os.system(f"rm -rf {self.data_output}/val_ind_{val_ind}{self.metrics_save_extra_name}_seed_{self.alternative_asimov_seed}_for_{data_type}")


  def _GetFiles(self, val_ind, data_type, force_sim=False):

    if not self.synth_vs_synth or force_sim:
      sim_file = [f"{self.val_file_loc}/val_ind_{val_ind}/{i}_{data_type}.parquet" for i in ["X","Y","wt"]]
    else:
      sim_file = [f"{self.data_output}/val_ind_{val_ind}{self.metrics_save_extra_name}_seed_{self.alternative_asimov_seed}_for_{data_type}/asimov.parquet"]
    if self.asimov_input is None:
      synth_file = [f"{self.data_output}/val_ind_{val_ind}{self.metrics_save_extra_name}_seed_{self.asimov_seed}_for_{data_type}/asimov.parquet"]
    else:
      synth_file = [f"{self.asimov_input}/val_ind_{val_ind}_seed_{self.asimov_seed}_for_{data_type}/asimov.parquet"]

    return sim_file, synth_file


  def DoLoss(self):

    # Get loss values
    for data_type in self.loss_datasets:
      if self.verbose:
        print(f"  - Getting loss for {data_type}")
      self.metrics[f"loss_{data_type}"] = self.network.Loss(
        f"{self.open_parameters['density']['file_loc']}/X_{data_type}.parquet", 
        f"{self.open_parameters['density']['file_loc']}/Y_{data_type}.parquet", 
        f"{self.open_parameters['density']['file_loc']}/wt_{data_type}.parquet"
      )


  def DoHistogramMetrics(self):

    # Loop through validation indices
    for val_ind, val_info in enumerate(GetValidationLoop(self.open_cfg, self.file_name)):
      if SkipNonDensity(self.open_cfg, self.file_name, val_info, skip_non_density=True): continue

      for data_type in self.histogram_datasets:
        if SkipEmptyDataset(self.open_cfg, self.file_name, data_type, val_info): continue

        sim_file, synth_file = self._GetFiles(val_ind, data_type)
        if not all([os.path.isfile(f) for f in sim_file]) or not all([os.path.isfile(f) for f in synth_file]): continue

        hm = HistogramMetrics(
          [sim_file], 
          [synth_file],
          self.open_cfg["variables"]
        )

        # Get chi squared values
        if self.do_chi_squared:
          if self.verbose:
            print(f" - Doing chi squared for {data_type} and val_ind {val_ind}")
          chi_squared, dof_for_chi_squared, chi_squared_per_dof = hm.GetChiSquared() 
          if len(chi_squared.keys()) != 0:
            self.metrics[f"chi_squared_{data_type}_val_ind_{val_ind}"] = chi_squared
            self.metrics[f"dof_for_chi_squared_{data_type}_val_ind_{val_ind}"] = dof_for_chi_squared
            self.metrics[f"chi_squared_per_dof_{data_type}_val_ind_{val_ind}"] = chi_squared_per_dof

        # Get KL divergence values
        if self.do_kl_divergence:
          if self.verbose:
            print(f" - Doing KL divergence for {data_type} and val_ind {val_ind}")
          kl_divergence = hm.GetKLDivergence()
          if len(kl_divergence.keys()) != 0:
            self.metrics[f"kl_divergence_{data_type}_val_ind_{val_ind}"] = kl_divergence

    # Get total values
    for data_type in self.histogram_datasets:
      if self.do_chi_squared:
        count_chi_squared_per_dof = 0
        chi_squared_per_dof_total = 0
        for val_ind, val_info in enumerate(GetValidationLoop(self.open_cfg, self.file_name)):
          if SkipNonDensity(self.open_cfg, self.file_name, val_info, skip_non_density=True): continue
          if SkipEmptyDataset(self.open_cfg, self.file_name, data_type, val_info): continue
          if f"chi_squared_per_dof_{data_type}_val_ind_{val_ind}" not in self.metrics: continue
          chi_squared_per_dof_total += self.metrics[f"chi_squared_per_dof_{data_type}_val_ind_{val_ind}"]["sum"]
          count_chi_squared_per_dof += len(self.open_cfg["variables"])
        self.metrics[f"chi_squared_per_dof_{data_type}_sum"] = chi_squared_per_dof_total
        if count_chi_squared_per_dof > 0:
          self.metrics[f"chi_squared_per_dof_{data_type}_mean"] = chi_squared_per_dof_total/count_chi_squared_per_dof

      if self.do_kl_divergence:
        count_kl_divergence = 0
        kl_divergence_total = 0
        for val_ind, val_info in enumerate(GetValidationLoop(self.open_cfg, self.file_name)):
          if SkipNonDensity(self.open_cfg, self.file_name, val_info, skip_non_density=True): continue
          if SkipEmptyDataset(self.open_cfg, self.file_name, data_type, val_info): continue
          if f"kl_divergence_{data_type}_val_ind_{val_ind}" not in self.metrics: continue
          kl_divergence_total += self.metrics[f"kl_divergence_{data_type}_val_ind_{val_ind}"]["sum"]
          count_kl_divergence += len(self.open_cfg["variables"])
        self.metrics[f"kl_divergence_{data_type}_sum"] = kl_divergence_total
        if count_kl_divergence > 0:
          self.metrics[f"kl_divergence_{data_type}_mean"] = kl_divergence_total/count_kl_divergence


  def DoMultiDimensionalDatasetMetrics(self):

    # Make yields for scaling down
    yield_class = Yields(
      self.open_parameters["yields"]["nominal"],
      lnN = self.open_parameters["yields"]["lnN"],
      physics_model = None,
      rate_param = f"mu_{self.file_name}" if f"mu_{self.file_name}" in self.open_cfg["inference"]["rate_parameters"] else None,
    )

    def scale_down(df, func, scale):
      df.loc[:,"wt"] /= scale/func(df).loc[:,"yield"]
      return df

    for data_type in self.multidimensional_datasets:
      if self.verbose:
        print(f"  - Getting multidimensional dataset metrics for {data_type}")

      # Get all files
      sim_files = []
      synth_files = []
      for val_ind, val_info in enumerate(GetValidationLoop(self.open_cfg, self.file_name)):
        if SkipNonDensity(self.open_cfg, self.file_name, val_info, skip_non_density=True): continue
        if SkipEmptyDataset(self.open_cfg, self.file_name, data_type, val_info): continue

        partial_scale_down = partial(scale_down, func=yield_class.GetYield, scale=self.open_parameters["eff_events"][data_type][val_ind])

        sim_file, synth_file = self._GetFiles(val_ind, data_type)
        if not all([os.path.isfile(f) for f in sim_file]) or not all([os.path.isfile(f) for f in synth_file]): continue
        sim_files.append(sim_file)
        synth_files.append(synth_file)

      # Initialise multi metrics
      mm = MultiDimMetrics(
        sim_files,
        synth_files,
        self.open_parameters['density'],
        functions_to_apply = [partial_scale_down]
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

      if self.do_kmeans_chi_squared:
        if self.verbose:
          print(" - Adding kmeans chi squared")
        mm.AddKMeansChiSquared()

      # Run metrics
      multidim_metrics = mm.Run()
      self.metrics = {**self.metrics, **{f"{k}_{data_type}": v for k, v in multidim_metrics.items()}}


  def DoInference(self):

    for data_type in self.inference_datasets:

      # Defaults in model
      defaults_in_model = {k:v for k,v in GetDefaultsInModel(self.file_name, self.open_cfg).items() if k in self.open_parameters["density"]["Y_columns"]}
      params_in_model = list(defaults_in_model.keys())

      # Build likelihood
      from likelihood import Likelihood
      lkld = Likelihood(
        {
          "pdfs" : {"inclusive":{self.open_parameters["file_name"] : self.network}},
        },
        likelihood_type = "unbinned",
        X_columns = self.open_parameters["density"]["X_columns"],
        Y_columns = params_in_model,
        Y_columns_per_model = {self.open_parameters["file_name"] : params_in_model},
        categories = ["inclusive"]
      )


      # Loop through validation indices
      for val_ind, val_info in enumerate(GetValidationLoop(self.open_cfg, self.file_name)):
        if SkipNonDensity(self.open_cfg, self.file_name, val_info, skip_non_density=True): continue
        if SkipEmptyDataset(self.open_cfg, self.file_name, data_type, val_info): continue

        if self.verbose:
          print(f"  - Running unbinned likelihood fit for the {data_type} dataset and Y:")
          print({k:v for k, v in val_info.items() if k in params_in_model})

        # Build test data loader
        sim_file = [f"{self.val_file_loc}/val_ind_{val_ind}/{i}_{data_type}.parquet" for i in ["X","Y","wt"]]
        scale = self.open_parameters["eff_events"][data_type][val_ind] / self.open_parameters["yields"]["nominal"]
        dps = DataProcessor(
          [sim_file],
          "parquet",
          wt_name = "wt",
          options = {
            "scale" : scale
          }
        )

        # Skip if empty
        if dps.GetFull(method="count") == 0: continue

        # Do initial fit
        lkld.GetBestFit({"inclusive":[dps]}, pd.DataFrame({k:[v] for k, v in defaults_in_model.items()}))

        # Add best fit values to metrics
        for col_index, col in enumerate(params_in_model):
          self.metrics[f"inference_best_fit_{data_type}_val_ind_{val_ind}_{col}"] = float(lkld.best_fit[col_index])

        # Get approximate uncertainty
        for col_index, col in enumerate(params_in_model):
          if self.verbose:
            print(f"  - Finding uncertainty estimates for {col}")
          uncert = lkld.GetApproximateUncertainty({"inclusive":[dps]}, col)
          true_value = float(val_info[col])
          if true_value > lkld.best_fit[col_index]:
            self.metrics[f"inference_chi_squared_{data_type}_val_ind_{val_ind}_{col}"] = float(((true_value - lkld.best_fit[col_index])**2) / (uncert[1]**2))
          else:
            self.metrics[f"inference_chi_squared_{data_type}_val_ind_{val_ind}_{col}"] = float(((true_value - lkld.best_fit[col_index])**2) / (uncert[-1]**2))
          self.metrics[f"inference_distance_{data_type}_val_ind_{val_ind}_{col}"] = abs(float(true_value - lkld.best_fit[col_index]))


      # Get chi squared values
      total_sum = 0.0
      total_count = 0.0
      for val_name, val_dict in self.metrics.items():
        if f"inference_chi_squared_{data_type}_val_ind_" in val_name:
          total_sum += val_dict
          total_count += 1
      if total_count > 0:
        self.metrics[f"inference_chi_squared_{data_type}"] = float(total_sum/total_count)

      # Get distance values
      total_sum = 0.0
      total_count = 0.0
      for val_name, val_dict in self.metrics.items():
        if f"inference_distance_{data_type}_val_ind_" in val_name:
          total_sum += val_dict
          total_count += 1
      if total_count > 0:
        self.metrics[f"inference_distance_{data_type}"] = float(total_sum/total_count)

      
  def Outputs(self):
    """
    Return a list of outputs given by class
    """

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add metrics
    outputs = [f"{self.data_output}/metrics{self.save_extra_name}{self.metrics_save_extra_name}.yaml"]

    # Add asimov
    if not (self.tidy_up_asimov or self.asimov_input is not None):
      dataset = []
      if self.do_histogram_metrics:
        dataset += self.histogram_datasets
      if self.do_multidimensional_dataset_metrics:
        dataset += self.multidimensional_datasets
      if self.do_inference:
        dataset += self.inference_datasets
      dataset = sorted(list(set(dataset)))

      for data_type in dataset:
        for val_ind, val_info in enumerate(GetValidationLoop(cfg, self.file_name)):
          if SkipNonDensity(cfg, self.file_name, val_info, skip_non_density=True): continue
          if SkipEmptyDataset(cfg, self.file_name, data_type, val_info): continue
          outputs += [f"{self.data_output}/val_ind_{val_ind}{self.metrics_save_extra_name}_seed_{self.asimov_seed}_for_{data_type}/asimov.parquet"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add parameters
    inputs += [self.parameters]

    # Open config
    cfg = LoadConfig(self.cfg)

    # Add density model
    density_model_name = f"{self.model_input}/{self.extra_model_dir}/{self.file_name}{self.save_extra_name}"
    density_model_name = density_model_name.replace("//", "/")
    inputs += [f"{density_model_name}.h5"]
    inputs += [f"{density_model_name}_architecture.yaml"]

    # Add data
    if self.do_loss:
      for data_type in self.loss_datasets:
        inputs += [f"{self.file_loc}/X_{data_type}.parquet"]
        inputs += [f"{self.file_loc}/Y_{data_type}.parquet"]
        inputs += [f"{self.file_loc}/wt_{data_type}.parquet"]

    datasets = []
    if self.do_histogram_metrics:
      datasets += self.histogram_datasets
    if self.do_multidimensional_dataset_metrics:
      datasets += self.multidimensional_datasets
    if self.do_inference:
      datasets += self.inference_datasets
    datasets = list(set(datasets))

    for data_type in datasets:
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, self.file_name)):
        if SkipNonDensity(cfg, self.file_name, val_info, skip_non_density=True): continue
        if SkipEmptyDataset(cfg, self.file_name, data_type, val_info): continue

        # Add input files
        inputs += [f"{self.val_file_loc}/val_ind_{val_ind}/{i}_{data_type}.parquet" for i in ["X","Y","wt"]]

      # Add premade asimov
      if self.asimov_input is not None:
        for val_ind, val_info in enumerate(GetValidationLoop(cfg, self.file_name)):
          if SkipNonDensity(cfg, self.file_name, val_info, skip_non_density=True): continue
          if SkipEmptyDataset(cfg, self.file_name, data_type, val_info): continue
          inputs += [f"{self.asimov_input}/val_ind_{val_ind}_seed_{self.asimov_seed}_for_{data_type}/asimov.parquet"]

    return inputs
