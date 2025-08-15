#!/usr/bin/env python3

import argparse
import os
import sys
import time
import yaml

import pyfiglet as pyg

from module import Module
from useful_functions import (
    CommonInferConfigOptions,
    GetBestFitFromYaml,
    GetBestFitWithShiftedNuisancesFromYaml,
    GetCategoryLoop,
    GetCombinedValdidationIndices,
    GetDataInput,
    GetDefaultsInModel,
    GetDictionaryEntryFromYaml,
    GetBaseFileLoop,
    GetFreezeLoop,
    GetModelLoop,
    GetModelFileLoop,
    GetParametersInModel,
    GetParameterLoop,
    GetScanArchitectures,
    GetValidationLoop,
    LoadConfig,
    OverwriteArchitecture,
    SetupSnakeMakeFile,
    SkipNonData,
    SkipNonDensity,
)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--asimov-seed', help='The seed to use the create the asimov', type=int, default=42)
  parser.add_argument('--benchmark', help='Run from benchmark scenario', default=None)
  parser.add_argument('--binned-fit-input', help='The inputs to do a binned fit either just bins ("X1[0,50,100,200]" or categories and bins "(X2<100):X1[0,50,100,200];(X2>100):X1[0,50,200]")', default=None)
  parser.add_argument('--cfg', help='Config for running', default=None)
  parser.add_argument('--custom-module', help='Name of custom module', default=None)
  parser.add_argument('--custom-options', help='Semi-colon separated list of options set by an equals sign to custom module', default="")
  parser.add_argument('--data-type', help='The data type to use when running the Generator, Bootstrap or Infer step. Default is sim for Bootstrap, and asimov for Infer.', type=str, default='sim', choices=['data', 'asimov', 'sim'])
  parser.add_argument('--density-architecture', help='Architecture for density model', type=str, default='configs/architecture/density_default.yaml')
  parser.add_argument('--density-performance-metrics', help='Comma separated list of density performance metrics', type=str, default='loss,histogram,multidim')
  parser.add_argument('--density-performance-metrics-multidim', help='Comma separated list of multidimensional density performance metrics', type=str, default='bdt,wasserstein,kmeans')
  parser.add_argument('--disable-tqdm', help='Disable tqdm when training.', action='store_true')
  parser.add_argument('--dry-run', help='Setup batch submission without running.', action='store_true')
  parser.add_argument('--extra-dir-name', help='Add extra name to step directory for input and output directory', type=str, default='')
  parser.add_argument('--extra-input-dir-name', help='Add extra name to step directory for input directory', type=str, default='')
  parser.add_argument('--extra-output-dir-name', help='Add extra name to step directory for output directory', type=str, default='')
  parser.add_argument('--extra-plot-name', help='Add extra name to infer step end of plot', type=str, default='')
  parser.add_argument('--extra-density-model-name', help='Add extra name to density model name', type=str, default='')
  parser.add_argument('--extra-regression-model-name', help='Add extra name to regression model name', type=str, default='')
  parser.add_argument('--freeze', help='Other inputs to likelihood and summary plotting', type=str, default=None)
  parser.add_argument('--hyperparameter-metric', help='Colon separated metric name and whether you want max or min, separated by a comma.', type=str, default='loss_test,min')
  parser.add_argument('--include-per-model-lnN', help='Include the lnN in the non-combined likelihood.', action='store_true')
  parser.add_argument('--include-per-model-rate', help='Include the rate parameters in the non-combined likelihood.', action='store_true')
  parser.add_argument('--include-postfit-uncertainty', help='Include the postfit uncertainties in the postfit plots.', action='store_true')
  parser.add_argument('--initial-best-fit-guess', help='The starting point of initial fit minimisation', default=None)
  parser.add_argument('--likelihood-type', help='Type of likelihood to use for fitting.', type=str, default='unbinned_extended', choices=['unbinned_extended', 'unbinned', 'binned_extended', 'binned'])
  parser.add_argument('--loop-over-epochs', help='Loop over epochs for performance metrics', action='store_true')
  parser.add_argument('--loop-over-lnN', help='Loop over log normal parameters as well as shape parameter', action='store_true')
  parser.add_argument('--loop-over-nuisances', help='Loop over nuisance parameters as well as POIs', action='store_true')
  parser.add_argument('--loop-over-rates', help='Loop over rate parameters as well as shape parameter', action='store_true')
  parser.add_argument('--make-snakemake-inputs', help='Make the snakemake input file', action='store_true')
  parser.add_argument('--minimisation-method', help='Method for minimisation', type=str, default='scipy')
  parser.add_argument('--number-of-asimov-events', help='The number of asimov events', type=int, default=10**6)
  parser.add_argument('--number-of-bootstraps', help='The number of bootstrap initial fits to run', type=int, default=100)
  parser.add_argument('--number-of-scan-points', help='The number of scan points run', type=int, default=41)
  parser.add_argument('--number-of-shuffles', help='The number of times to loop through the dataset when shuffling in preprocess', type=int, default=10)
  parser.add_argument('--number-of-toys', help='The number of toys for p-value dataset comparisons', type=int, default=100)
  parser.add_argument('--number-of-trials', help='The number of trials to test for BayesianHyperparameterTuning', type=int, default=10)
  parser.add_argument('--no-constraint', help='Do not use the constraints', action='store_true')
  parser.add_argument('--only-density', help='Build asimov from only the density model', action='store_true')
  parser.add_argument('--other-input', help='Other inputs to likelihood and summary plotting', type=str, default=None)
  parser.add_argument('--overwrite-density-architecture', help='Comma separated list of key=values to overwrite density architecture parameters', type=str, default='')
  parser.add_argument('--overwrite-regression-architecture', help='Comma separated list of key=values to overwrite regression architecture parameters', type=str, default='')
  parser.add_argument('--plot-2d-unrolled', help='Make 2D unrolled plots when running generator.', action='store_true')
  parser.add_argument('--plot-transformed', help='Plot transformed variables when running generator.', action='store_true')
  parser.add_argument('--points-per-job', help='The number of points ran per job', type=int, default=1)
  parser.add_argument('--prefit-nuisance-values', help='Make postfit plots with prefit nuisance values', action='store_true')
  parser.add_argument('--quiet', help='No verbose output.', action='store_true')
  parser.add_argument('--regression-architecture', help='Architecture for regression model', type=str, default='configs/architecture/regression_default.yaml')
  parser.add_argument('--replace-inputs', help='Colon and comma separated string to replace the inputs', type=str, default=None)
  parser.add_argument('--replace-outputs', help='Colon and comma separated string to replace the outputs and write a dummy file', type=str, default=None)
  parser.add_argument('--save-model-per-epoch', help='Save a model at each epoch', action='store_true')
  parser.add_argument('--scale-to-eff-events', help='Scale to the number of effective events rather than the yield.', action='store_true')
  parser.add_argument('--sigma-between-scan-points', help='The estimated unprofiled sigma between the scanning points', type=float, default=0.2)
  parser.add_argument('--sim-type', help='The split of simulated data to use with Infer.', type=str, default='val')
  parser.add_argument('--skip-non-density', help='Skip the validation points that are not in the validation model', action='store_true')
  parser.add_argument('--snakemake-cfg', help='Config for running with snakemake', default=None)
  parser.add_argument('--snakemake-dry-run', help='Dry run snakemake', action='store_true')
  parser.add_argument('--snakemake-force', help='Force snakemake to execute all steps', action='store_true')
  parser.add_argument('--snakemake-force-local', help='Force step to execute locally when running snakemake', action='store_true')
  parser.add_argument('--specific', help='Specific part of a step to run.', type=str, default='')
  parser.add_argument('--step', help='Step to run.', type=str, default=None)
  parser.add_argument('--submit', help='Batch to submit to', type=str, default=None)
  parser.add_argument('--summary-from', help='Summary from bootstrap or likelihood scan', type=str, default='Covariance', choices=['Scan', 'Bootstrap','ApproximateUncertainty','Covariance','CovarianceWithDMatrix'])
  parser.add_argument('--summary-nominal-name', help='Name of nominal summary points', type=str, default='Nominal')
  parser.add_argument('--summary-show-2sigma', help='Show 2 sigma band on the summary.', action='store_true')
  parser.add_argument('--summary-show-chi-squared', help='Add the chi squared value to the plot', action='store_true')
  parser.add_argument('--summary-subtract', help='Use subtraction instead of division in summary', action='store_true')
  parser.add_argument('--use-asimov-scaling', help='Generate asimov with this scaling up of the predicted yield', type=int, default=10)
  parser.add_argument('--use-expected-data-uncertainty', help='In postfit plots change the data uncertainty to the expected stat uncertainty', action='store_true')
  parser.add_argument('--use-wandb', help='Use wandb for logging.', action='store_true')
  parser.add_argument('--val-inds', help='val_inds for summary plots.', type=str, default=None)
  parser.add_argument('--wandb-project-name', help='Name of project on wandb', type=str, default='innfer')
  default_args = parser.parse_args([])
  args = parser.parse_args()

  # Check inputs
  if args.cfg is None and args.benchmark is None:
    raise ValueError("The --cfg or --benchmark is required.")
  if args.step is None:
    raise ValueError("The --step is required.")
  if args.data_type != "sim" and args.scale_to_eff_events:
    raise ValueError("The --scale-to-eff-events option is only valid for --data-type=sim.")

  # Adjust input paths
  if os.path.exists(f"configs/run/{args.cfg}"): # Change cfg path
    args.cfg = f"configs/run/{args.cfg}"
  if os.path.exists(f"configs/architecture/{args.density_architecture}"): # Change architecture path
    args.density_architecture = f"configs/architecture/{args.density_architecture}"
  if os.path.exists(f"configs/architecture/{args.regression_architecture}"): # Change architecture path
    args.regression_architecture = f"configs/architecture/{args.regression_architecture}"
  if args.snakemake_cfg is not None:
    if os.path.exists(f"configs/snakemake/{args.snakemake_cfg}"): # Change snakemake cfg path
      args.snakemake_cfg = f"configs/snakemake/{args.snakemake_cfg}"
  if args.submit is not None: # Change submit path
    if os.path.exists(f"configs/submit/{args.submit}"):
      args.submit = f"configs/submit/{args.submit}"
  if args.benchmark is not None: # Change benchmark path
    if os.path.exists(f"configs/run/{args.benchmark}") and ".yaml" in args.benchmark:
      args.benchmark = f"configs/run/{args.benchmark}"
  if args.step != "MakeBenchmark" and args.benchmark is not None: # Set cfg name for benchmark scenarios
      args.cfg = f"configs/run/Benchmark_{args.benchmark}.yaml"
  if args.submit is not None:
    args.disable_tqdm = True

  # Check if the density architecture is benchmark
  if args.density_architecture == "Benchmark" or args.step == "SetupDensityFromBenchmark":
    if args.step != "SetupDensityFromBenchmark":
      print("WARNING: Make sure you run SetupDensityFromBenchmark before running the other steps when using density_architecture=Benchmark.")

  # Overwrite architecture
  if args.overwrite_density_architecture != "":
    args.density_architecture = OverwriteArchitecture(args.density_architecture, args.overwrite_density_architecture)
  if args.overwrite_regression_architecture != "":
    args.regression_architecture = OverwriteArchitecture(args.regression_architecture, args.overwrite_regression_architecture)

  return args, default_args


def main(args, default_args):

  # Get extra input and output directory name
  if args.extra_dir_name != "":
    args.extra_input_dir_name = args.extra_dir_name
    args.extra_output_dir_name = args.extra_dir_name

  # Initiate module class
  module = Module(
    sys.argv,
    args,
    default_args,
  )


  # Make the benchmark scenario
  if args.step == "MakeBenchmark":
    print("<< Making benchmark inputs >>")
    module.job_name = f"jobs/Benchmark_{args.benchmark}/innfer_{args.step}" if ".yaml" not in args.benchmark else f"jobs/Benchmark_{args.benchmark.split('/')[-1].split('.yaml')[0]}/innfer_{args.step}"
    module.Run(
      module_name = "make_benchmark",
      class_name = "MakeBenchmark",
      config = {
        "name" : args.benchmark,
        "verbose" : not args.quiet,
      },
      loop = {},
      force = True,
    )
  else:
    # Load in configuration file
    cfg = LoadConfig(args.cfg)
    module.job_name = f"jobs/{cfg['name']}/innfer_{args.step}"
    # Set up output directories
    data_dir = f"{os.getenv('DATA_DIR')}/{cfg['name']}"
    plots_dir = f"{os.getenv('PLOTS_DIR')}/{cfg['name']}"
    models_dir = f"{os.getenv('MODELS_DIR')}/{cfg['name']}"


  # Prepare the dataset
  if args.step == "LoadData":
    print("<< Load data in to make the base datasets >>")
    for base_file_name in GetBaseFileLoop(cfg):
      module.Run(
        module_name = "load_data",
        class_name = "LoadData",
        config = {
          "cfg" : args.cfg,
          "file_name" : base_file_name,
          "data_output" : f"{data_dir}/LoadData{args.extra_output_dir_name}",
          "number_of_shuffles" : args.number_of_shuffles,
          "verbose" : not args.quiet,
        },
        loop = {"base_file_name" : base_file_name},
      )


  # PreProcess the dataset
  if args.step == "PreProcess":
    print("<< Preprocessing datasets for training, testing and validation >>")
    for file_name in GetModelFileLoop(cfg):
      for category in GetCategoryLoop(cfg):
        module.Run(
          module_name = "preprocess",
          class_name = "PreProcess",
          config = {
            "cfg" : args.cfg,
            "file_name" : file_name,
            "data_input" : f"{data_dir}/LoadData",
            "data_output" : f"{data_dir}/PreProcess/{file_name}/{category}",
            "number_of_shuffles" : args.number_of_shuffles,
            "extra_selection" : cfg["categories"][category] if "categories" in cfg and category in cfg["categories"] and cfg["categories"][category] != "inclusive" else None,
            "category" : category,
            "verbose" : not args.quiet,
          },
          loop = {"file_name" : file_name, "category" : category},
        )


  # Create a "toy" dataset from oversampling the validation data for 0
  if args.step == "ResampleValidationForData":
    module.Run(
      module_name = "resample_validation_for_data",
      class_name = "ResampleValidationForData",
      config = {
        "data_output" : cfg["data_file"],
        "data_inputs" : {file_name : {category:f"{data_dir}/PreProcess/{file_name}/{category}/val_ind_0" for category in GetCategoryLoop(cfg)} for file_name in GetModelFileLoop(cfg)},
        "verbose" : not args.quiet,
      },
      loop = {},
    )


  # Make data categories
  if args.step == "DataCategories":
    print("<< Making data categories >>")
    for category in GetCategoryLoop(cfg):
      module.Run(
        module_name = "data_categories",
        class_name = "DataCategories",
        config = {
          "selection" : cfg["data_selection"],
          "extra_selection" : cfg["categories"][category] if "categories" in cfg and category in cfg["categories"] and cfg["categories"][category] != "inclusive" else None,
          "data_input" : cfg["data_file"],
          "add_columns" : cfg["data_add_columns"],
          "data_output" : f"{data_dir}/DataCategories/{category}",
          "verbose" : not args.quiet,
        },
        loop = {"category" : category},
      )


  # Custom
  if args.step == "Custom":
    print("<< Custom module >>")
    module.Run(
      module_name = args.custom_module,
      class_name = args.custom_module,
      config = {
        "cfg" : args.cfg,
        "options" : {i.split("=")[0] : i.split("=")[1] for i in (args.custom_options.split(";") if args.custom_options != "" else [])},
      },
      loop = {},
    )


  # Plot preprocessed training and testing data 
  if args.step == "InputPlotTraining":
    print("<< Plotting training and testing datasets >>")
    for model_info in GetModelLoop(cfg):
      module.Run(
        module_name = "input_plot_training",
        class_name = "InputPlotTraining",
        config = {
          "cfg" : args.cfg,
          "parameters" : model_info["parameters"],
          "data_input" : model_info['file_loc'],
          "plots_output" : f"{plots_dir}/InputPlotTraining{args.extra_output_dir_name}/{model_info['name']}",
          "model_type" : model_info['type'],
          "file_name" : model_info['file_name'],
          "parameter" : model_info["parameter"],
          "split" : model_info['split'],
          "verbose" : not args.quiet,
        },
        loop = {"model_name" : model_info['name']},
      )


  # Plot preprocessed validation data
  if args.step == "InputPlotValidation":
    print("<< Plotting validation datasets >>")
    for file_name in GetModelFileLoop(cfg):
      for category in GetCategoryLoop(cfg):
        module.Run(
          module_name = "input_plot_validation",
          class_name = "InputPlotValidation",
          config = {
            "cfg" : args.cfg,
            "file_name" : file_name,
            "parameters" : f"{data_dir}/PreProcess/{file_name}/{category}/parameters.yaml",
            "data_input" : f"{data_dir}/PreProcess/{file_name}/{category}",
            "plots_output" : f"{plots_dir}/InputPlotValidation{args.extra_input_dir_name}/{file_name}/{category}",
            "val_loop" : GetValidationLoop(cfg, file_name),
            "verbose" : not args.quiet,
          },
          loop = {"file_name" : file_name, "category" : category},
        )


  # Train density network
  if args.step == "TrainDensity":
    print("<< Training the density networks >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      module.Run(
        module_name = "train_density",
        class_name = "TrainDensity",
        config = {
          "parameters" : model_info["parameters"],
          "architecture" : args.density_architecture,
          "file_name" : model_info["file_name"],
          "data_input" : model_info['file_loc'],
          "data_output" : f"{models_dir}/{model_info['name']}{args.extra_density_model_name}",
          "plots_output" : f"{plots_dir}/TrainDensity/{model_info['name']}{args.extra_density_model_name}",
          "disable_tqdm" : args.disable_tqdm,
          "use_wandb" : args.use_wandb,
          "initiate_wandb" : args.use_wandb,
          "wandb_project_name" : args.wandb_project_name,
          "wandb_submit_name" : f"{cfg['name']}_{model_info['name']}{args.extra_density_model_name}",
          "save_model_per_epoch" : args.save_model_per_epoch,
          "verbose" : not args.quiet,        
        },
        loop = {"model_name" : model_info['name']}
      )


  # Setup density from benchmark
  if args.step == "SetupDensityFromBenchmark":
    print("<< Setup density from benchmark >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      module.Run(
        module_name = "setup_density_from_benchmark",
        class_name = "SetupDensityFromBenchmark",
        config = {
          "cfg" : args.cfg,
          "file_name" : model_info["file_name"],
          "benchmark" : args.benchmark,
          "data_output" : f"{models_dir}/{model_info['name']}{args.extra_density_model_name}",
          "verbose" : not args.quiet,        
        },
        loop = {"model_name" : model_info['name']}
      )


  # Train regression network
  if args.step == "TrainRegression":
    print("<< Training the regression networks >>")
    for model_info in GetModelLoop(cfg, only_regression=True):
      module.Run(
        module_name = "train_regression",
        class_name = "TrainRegression",
        config = {
          "parameters" : model_info["parameters"],
          "architecture" : args.regression_architecture,
          "file_name" : model_info["file_name"],
          "data_input" : model_info['file_loc'],
          "parameter" : model_info["parameter"],
          "data_output" : f"{models_dir}/{model_info['name']}{args.extra_regression_model_name}",
          "plots_output" : f"{plots_dir}/TrainRegression/{model_info['name']}{args.extra_regression_model_name}",
          "disable_tqdm" : args.disable_tqdm,
          "use_wandb" : args.use_wandb,
          "initiate_wandb" : args.use_wandb,
          "wandb_project_name" : args.wandb_project_name,
          "wandb_submit_name" : f"{cfg['name']}_{model_info['name']}",
          "save_model_per_epoch" : args.save_model_per_epoch,
          "verbose" : not args.quiet,        
        },
        loop = {"model_name" : model_info['name']}
      )


  # Evaluate the regression models
  if args.step == "EvaluateRegression":
    print("<< Evaluating the regression networks >>")
    for model_info in GetModelLoop(cfg, only_regression=True):
      module.Run(
        module_name = "evaluate_regression",
        class_name = "EvaluateRegression",
        config = {
          "data_input" : model_info['file_loc'],
          "plots_output" : f"{plots_dir}/EvaluateRegression/{model_info['file_name']}{args.extra_regression_model_name}",
          "model_input" : f"{models_dir}",
          "model_name" : f"{model_info['name']}{args.extra_regression_model_name}",
          "file_name" : model_info["file_name"],
          "parameters" : model_info["parameters"],
          "parameter" : model_info["parameter"],
          "data_output" : f"{data_dir}/EvaluateRegression/{model_info['name']}{args.extra_regression_model_name}",
          "verbose" : not args.quiet,        
        },
        loop = {"model_name" : model_info['name']}
      )


  # Plot the regression models
  if args.step == "PlotRegression":
    print("<< Plotting the regression distributions >>")
    for model_info in GetModelLoop(cfg, only_regression=True):
      module.Run(
        module_name = "plot_regression",
        class_name = "PlotRegression",
        config = {
          "cfg" : args.cfg,
          "data_input" : model_info['file_loc'],
          "model_name" : f"{model_info['name']}{args.extra_regression_model_name}",
          "parameters" : model_info["parameters"],
          "parameter" : model_info["parameter"],
          "evaluate_input" : f"{data_dir}/EvaluateRegression/{model_info['name']}{args.extra_regression_model_name}",
          "plots_output" : f"{plots_dir}/PlotRegression/{model_info['name']}{args.extra_regression_model_name}",
          "verbose" : not args.quiet,        
        },
        loop = {"model_name" : model_info['name']}
      )


  # Make the asimov datasets
  if args.step == "MakeAsimov":
    print(f"<< Making the asimov datasets >>")
    for file_name in GetModelFileLoop(cfg):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        for category in GetCategoryLoop(cfg):
          module.Run(
            module_name = "make_asimov",
            class_name = "MakeAsimov",
            config = {
              "cfg" : args.cfg,
              "density_model" : GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=category)[0],
              "regression_models" : GetModelLoop(cfg, model_file_name=file_name, only_regression=True, specific_category=category),
              "regression_spline_input" : f"{data_dir}/EvaluateRegression",
              "model_input" : f"{models_dir}",
              "extra_density_model_name" : args.extra_density_model_name,
              "extra_regression_model_name" : args.extra_regression_model_name,
              "parameters" : f"{data_dir}/PreProcess/{file_name}/{category}/parameters.yaml",
              "data_output" : f"{data_dir}/MakeAsimov{args.extra_output_dir_name}/{file_name}/{category}/val_ind_{val_ind}",
              "n_asimov_events" : args.number_of_asimov_events,
              "seed" : args.asimov_seed,
              "val_info" : val_info,
              "val_ind" : val_ind,
              "only_density" : args.only_density,
              "verbose" : not args.quiet,
              "file_name" : file_name,
              "use_asimov_scaling" : args.use_asimov_scaling,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "category" : category},
          )


  # Get performance metrics
  if args.step == "DensityPerformanceMetrics":
    print("<< Getting the performance metrics of the trained networks >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      for extra_name in [f"_epoch_{i}" for i in range(GetDictionaryEntryFromYaml(f"{models_dir}/{model_info['name']}{args.extra_density_model_name}/{model_info['file_name']}_architecture.yaml", ["epochs"])+1)] if args.loop_over_epochs else [""]:
        module.Run(
          module_name = "density_performance_metrics",
          class_name = "DensityPerformanceMetrics",
          config = {
            "cfg" : args.cfg,
            "file_name" : model_info["file_name"],
            "parameters" : model_info["parameters"],
            "file_loc" : model_info['file_loc'],
            "val_file_loc" : model_info['val_file_loc'],
            "model_input" : f"{models_dir}",
            "extra_model_dir" : f"{model_info['name']}{args.extra_density_model_name}",
            "data_output" : f"{data_dir}/DensityPerformanceMetrics{args.extra_output_dir_name}/{model_info['name']}{args.extra_density_model_name}",
            "do_inference": "inference" in args.density_performance_metrics,
            "do_loss": "loss" in args.density_performance_metrics,
            "do_histogram_metrics": "histogram" in args.density_performance_metrics,
            "do_multidimensional_dataset_metrics": "multidim" in args.density_performance_metrics,
            "do_bdt_separation" : "bdt" in args.density_performance_metrics_multidim,
            "do_wasserstein" : "wasserstein" in args.density_performance_metrics_multidim,
            "do_sliced_wasserstein" : "wasserstein" in args.density_performance_metrics_multidim,
            "do_kmeans_chi_squared" : "kmeans" in args.density_performance_metrics_multidim,
            "save_extra_name": extra_name,
            "n_asimov_events" : args.number_of_asimov_events,
            "seed" : args.asimov_seed,
            "verbose" : not args.quiet,     
          },
          loop = {"model_name" : model_info['name'], "extra_name" : extra_name}
        )


  # Plot performance metrics per epoch
  if args.step == "EpochPerformanceMetricsPlot":
    print("<< Plotting the performance metrics as a function of the epoch of training >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      module.Run(
        module_name = "epoch_performance_metrics_plot",
        class_name = "EpochPerformanceMetricsPlot",
        config = {
          "architecture": f"{models_dir}/{model_info['name']}{args.extra_density_model_name}/{model_info['file_name']}_architecture.yaml",
          "data_input" : f"{data_dir}/DensityPerformanceMetrics{args.extra_input_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "plots_output" : f"{plots_dir}/EpochPerformanceMetricsPlot{args.extra_output_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "merged_plot" : args.other_input.split(",") if args.other_input is not None else None,
          "verbose" : not args.quiet,  
        },
        loop = {"model_name" : model_info['name']}
      )


  # Perform a p-value dataset comparison test for sim vs synth
  if args.step == "PValueSimVsSynth":
    print("<< Getting the metrics for Sim Vs Synth comparison >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      module.Run(
        module_name = "density_performance_metrics",
        class_name = "DensityPerformanceMetrics",
        config = {
          "cfg" : args.cfg,
          "file_name" : model_info["file_name"],
          "parameters" : model_info["parameters"],
          "model_input" : f"{models_dir}",
          "extra_model_dir" : f"{model_info['name']}{args.extra_density_model_name}",
          "file_loc" : model_info['file_loc'],
          "val_file_loc" : model_info['val_file_loc'],
          "data_output" : f"{data_dir}/PValueSimVsSynth{args.extra_output_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "do_inference": False,
          "do_loss": False,
          "do_histogram_metrics": False,
          "do_multidimensional_dataset_metrics": True,
          "do_bdt_separation" : "bdt" in args.density_performance_metrics_multidim,
          "do_wasserstein" : "wasserstein" in args.density_performance_metrics_multidim,
          "do_sliced_wasserstein" : "wasserstein" in args.density_performance_metrics_multidim,
          "do_kmeans_chi_squared" : "kmeans" in args.density_performance_metrics_multidim,
          "n_asimov_events" : args.number_of_asimov_events,
          "seed" : args.asimov_seed,
          "use_eff_events" : True,
          "verbose" : not args.quiet,     
        },
        loop = {"model_name" : model_info['name']}
      )


  # Perform a p-value dataset comparison test bootstrapping synth vs synth
  if args.step == "PValueSynthVsSynth":
    print("<< Running the distributions of bootstrapped Synth Vs Synth >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      for toy in range(args.number_of_toys):
        module.Run(
          module_name = "density_performance_metrics",
          class_name = "DensityPerformanceMetrics",
          config = {
            "cfg" : args.cfg,
            "file_name" : model_info["file_name"],
            "parameters" : model_info["parameters"],
            "model_input" : f"{models_dir}",
            "extra_model_dir" : f"{model_info['name']}{args.extra_density_model_name}",
            "file_loc" : model_info['file_loc'],
            "val_file_loc" : model_info['val_file_loc'],
            "data_output" : f"{data_dir}/PValueSynthVsSynth{args.extra_output_dir_name}/{model_info['name']}{args.extra_density_model_name}",
            "do_inference": False,
            "do_loss": False,
            "do_histogram_metrics": False,
            "do_multidimensional_dataset_metrics": True,
            "do_bdt_separation" : "bdt" in args.density_performance_metrics_multidim,
            "do_wasserstein" : "wasserstein" in args.density_performance_metrics_multidim,
            "do_sliced_wasserstein" : "wasserstein" in args.density_performance_metrics_multidim,
            "do_kmeans_chi_squared" : "kmeans" in args.density_performance_metrics_multidim,
            "n_asimov_events" : args.number_of_asimov_events,
            "seed" : args.asimov_seed,
            "synth_vs_synth" : True,
            "alternative_asimov_seed_shift" : toy,
            "metrics_save_extra_name" : f"_toy_{toy}",
            "asimov_input" : f"{data_dir}/PValueSimVsSynth{args.extra_input_dir_name}/{model_info['name']}{args.extra_density_model_name}",
            "use_eff_events" : True,
            "verbose" : not args.quiet,
          },
          loop = {"model_name" : model_info['name'], "toy" : toy}
        )


  # Collect the synth vs synth tests
  if args.step == "PValueSynthVsSynthCollect":
    print("<< Collecting the classifier 2 sample test >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      module.Run(
        module_name = "p_value_synth_vs_synth_collect",
        class_name = "PValueSynthVsSynthCollect",
        config = {
          "data_input" : f"{data_dir}/PValueSynthVsSynth{args.extra_input_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "data_output" : f"{data_dir}/PValueSynthVsSynthCollect{args.extra_output_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "number_of_toys" : args.number_of_toys,
          "verbose" : not args.quiet,  
        },
        loop = {"model_name" : model_info['name']}
      )


  # Plot the p values dataset comparisons
  if args.step == "PValueDatasetComparisonPlot":
    print("<< Plotting p-value dataset comparisons >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      module.Run(
        module_name = "p_value_dataset_comparison_plot",
        class_name = "PValueDatasetComparisonPlot",
        config = {
          "synth_vs_synth_input" : f"{data_dir}/PValueSynthVsSynthCollect{args.extra_input_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "sim_vs_synth_input" : f"{data_dir}/PValueSimVsSynth{args.extra_input_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "plots_output" : f"{plots_dir}/PValueDatasetComparisonPlot{args.extra_output_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "verbose" : not args.quiet,  
        },
        loop = {"model_name" : model_info['name']}
      )


  # Perform a hyperparameter scan
  if args.step == "HyperparameterScan":
    print("<< Running a hyperparameter scan >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      for architecture_ind, architecture in enumerate(GetScanArchitectures(args.density_architecture, data_output=f"{data_dir}/HyperparameterScan/{model_info['name']}/")):
        module.Run(
          module_name = "hyperparameter_scan",
          class_name = "HyperparameterScan",
          config = {
            "cfg" : args.cfg,
            "data_input" : f"{data_dir}/PreProcess{args.extra_input_dir_name}",
            "parameters" : model_info["parameters"],
            "architecture" : architecture,
            "file_name" : model_info["file_name"],
            "data_output" : f"{data_dir}/HyperparameterScan{args.extra_output_dir_name}/{model_info['name']}{args.extra_density_model_name}",
            "use_wandb" : args.use_wandb,
            "wandb_project_name" : args.wandb_project_name,
            "wandb_submit_name" : f"{cfg['name']}_{model_info['name']}{args.extra_density_model_name}",
            "disable_tqdm" : args.disable_tqdm,
            "save_extra_name" : f"_{architecture_ind}",
            "density_performance_metrics" : args.density_performance_metrics,
            "verbose" : not args.quiet,        
          },
          loop = {"model_name" : model_info['name'], "architecture_ind" : architecture_ind}
        )


  # Collect a hyperparameter scan
  if args.step == "HyperparameterScanCollect":
    print("<< Collecting hyperparameter scan >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      module.Run(
        module_name = "hyperparameter_scan_collect",
        class_name = "HyperparameterScanCollect",
        config = {
          "file_name" : model_info["file_name"],
          "data_input" : f"{data_dir}/HyperparameterScan{args.extra_input_dir_name}/{model_info['name']}{args.extra_density_model_name}",
          "data_output" : f"{models_dir}/{model_info['name']}{args.extra_density_model_name}",
          "save_extra_names" : [f"_{architecture_ind}" for architecture_ind in range(len(GetScanArchitectures(args.density_architecture, write=False)))],
          "metric" : args.hyperparameter_metric,
          "verbose" : not args.quiet,        
        },
        loop = {"model_name" : model_info['name']}
      )


  # Perform a hyperparameter scan
  if args.step == "BayesianHyperparameterTuning":
    print("<< Running a bayesian hyperparameter tuning >>")
    for model_info in GetModelLoop(cfg, only_density=True):
      module.Run(
        module_name = "bayesian_hyperparameter_tuning",
        class_name = "BayesianHyperparameterTuning",
        config = {
            "cfg" : args.cfg,
            "data_input" : f"{data_dir}/PreProcess",
            "parameters" : model_info["parameters"],
            "tune_architecture" : args.density_architecture,
            "file_name" : model_info["file_name"],
            "file_loc" : model_info['file_loc'],
            "val_file_loc" : model_info['val_file_loc'],
            "best_model_output" : f"{models_dir}/{model_info['name']}{args.extra_density_model_name}",
            "data_output" : f"{data_dir}/BayesianHyperparameterTuning{args.extra_output_dir_name}/{model_info['name']}{args.extra_density_model_name}",
            "use_wandb" : args.use_wandb,
            "wandb_project_name" : args.wandb_project_name,
            "wandb_submit_name" : f"{cfg['name']}_{model_info['name']}{args.extra_density_model_name}",
            "disable_tqdm" : args.disable_tqdm,
            "density_performance_metrics" : args.density_performance_metrics,
            "n_trials" : args.number_of_trials,
            "metric" : args.hyperparameter_metric,
            "verbose" : not args.quiet,     
        },
        loop = {"model_name" : model_info['name']}
      )


  # Making plots using the network as a generator for individual Y values
  if args.step == "Flow":
    print("<< Making plots looking at the flow of the network >>")
    for file_name in GetModelFileLoop(cfg):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=True): continue
        for category in GetCategoryLoop(cfg):
          module.Run(
            module_name = "flow",
            class_name = "Flow",
            config = {
              "model_input" : f"{models_dir}",
              "density_model" : GetModelLoop(cfg, model_file_name=file_name, only_density=True, specific_category=category)[0],
              "data_input" : f"{data_dir}/PreProcess/{file_name}/{category}/val_ind_{val_ind}",
              "plots_output" : f"{plots_dir}/Flow{args.extra_output_dir_name}/{file_name}/{category}",
              "extra_plot_name" : f"{val_ind}_{args.extra_plot_name}" if args.extra_plot_name != "" else str(val_ind),
              "sim_type" : args.sim_type,
              "verbose" : not args.quiet,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "category" : category},
          )


  # Making plots using the network as a generator for individual Y values
  if args.step == "Generator":
    print("<< Making plots using the network as a generator for individual Y values >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        for category in GetCategoryLoop(cfg):
          module.Run(
            module_name = "generator",
            class_name = "Generator",
            config = {
              "cfg" : args.cfg,
              "data_input" : GetDataInput("sim", cfg, file_name, val_ind, data_dir, sim_type=args.sim_type)[category],
              "asimov_input": GetDataInput("asimov", cfg, file_name, val_ind, data_dir, asimov_dir_name=f"MakeAsimov{args.extra_input_dir_name}")[category],
              "plots_output" : f"{plots_dir}/Generator{args.extra_output_dir_name}/{file_name}/{category}",
              "do_2d_unrolled" : args.plot_2d_unrolled,
              "extra_plot_name" : f"{val_ind}_{args.extra_plot_name}" if args.extra_plot_name != "" else str(val_ind),
              "sim_type" : args.sim_type,
              "val_info" : val_info,
              "plot_styles" : [2],
              "no_text" : False,
              "data_label" : "Simulated",
              "stack_label" : "Synthetic",
              "verbose" : not args.quiet,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "category" : category},
          )


  # Making a summary plot using the network as a generator for individual Y values
  if args.step == "GeneratorSummary":
    print("<< Making plots using the network as a generator summarising all Y values >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      validation_loop = GetValidationLoop(cfg, file_name)
      for category in GetCategoryLoop(cfg):
        module.Run(
          module_name = "generator_summary",
          class_name = "GeneratorSummary",
          config = {
            "cfg" : args.cfg,
            "val_loop" : validation_loop,
            "data_input" : [{k:f"{data_dir}/PreProcess/{k}/{category}/val_ind_{v}" for k,v in GetCombinedValdidationIndices(cfg, file_name, val_ind).items()} for val_ind in range(len(validation_loop))],
            "asimov_input": [{k:f"{data_dir}/MakeAsimov{args.extra_input_dir_name}/{k}/{category}/val_ind_{v}" for k,v in GetCombinedValdidationIndices(cfg, file_name, val_ind).items()} for val_ind in range(len(validation_loop))],
            "sim_type" : args.sim_type,
            "plots_output" : f"{plots_dir}/GeneratorSummary{args.extra_output_dir_name}/{file_name}/{category}",
            "extra_plot_name" : args.extra_plot_name,
            "file_name" : file_name,
            "val_inds" : args.val_inds,
            "verbose" : not args.quiet,
          },
          loop = {"file_name" : file_name, "category" : category},
        )


  # Run likelihood debug
  if args.step == "LikelihoodDebug":
    print(f"<< Running a single likelihood value for Y={args.other_input} >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
            "method" : "Debug",
            "extra_file_name" : str(val_ind),
            "other_input" : args.other_input,
            "val_ind" : val_ind
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )


  # Run initial fits from a full dataset
  if args.step == "InitialFit":
    print(f"<< Running initial fits >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
              "method" : "InitialFit",
              "data_output" : f"{data_dir}/InitialFit{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
              "extra_file_name" : str(val_ind),
              "freeze" : freeze["freeze"],
              "val_ind" : val_ind,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "freeze_ind" : freeze_ind},
          )


  # Run approximate uncertainties
  if args.step == "ApproximateUncertainty":
    print(f"<< Finding the approximate uncertainties >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for column in GetParameterLoop(file_name, cfg, include_nuisances=args.loop_over_nuisances, include_rate=args.loop_over_rates, include_lnN=args.loop_over_lnN):
          for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
            module.Run(
              module_name = "infer",
              class_name = "Infer",
              config = {
                **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
                "method" : "ApproximateUncertainty",
                "best_fit_input" : f"{data_dir}/InitialFit{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
                "data_output" : f"{data_dir}/ApproximateUncertainty{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
                "column" : column,
                "extra_file_name" : str(val_ind),
                "freeze" : freeze["freeze"],
                "val_ind" : val_ind,
              },
              loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column, "freeze_ind" : freeze_ind},
            )


  # Get the Hessian matrix
  if args.step == "Hessian" and args.likelihood_type in ["unbinned", "unbinned_extended"]:
    print(f"<< Calculating the Hessian matrix >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
              "method" : "Hessian",
              "best_fit_input" : f"{data_dir}/InitialFit{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
              "data_output" : f"{data_dir}/Hessian{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
              "extra_file_name" : str(val_ind),
              "freeze" : freeze["freeze"],
              "val_ind" : val_ind,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "freeze_ind" : freeze_ind},
          )


  # Get the Hessian matrix running in parallel
  if args.step == "HessianParallel":
    print(f"<< Calculating the Hessian matrix in parallel >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
          columns = [col for col in GetDefaultsInModel(file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN).keys() if col not in freeze["freeze"].keys()]
          for column_1_ind, column_1 in enumerate(columns):
            for column_2_ind, column_2 in enumerate(columns):
              if column_1_ind > column_2_ind: continue
              module.Run(
                module_name = "infer",
                class_name = "Infer",
                config = {
                  **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
                  "method" : "HessianParallel",
                  "best_fit_input" : f"{data_dir}/InitialFit{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
                  "data_output" : f"{data_dir}/HessianParallel{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
                  "extra_file_name" : str(val_ind),
                  "freeze" : freeze["freeze"],
                  "val_ind" : val_ind,
                  "hessian_parallel_column_1" : column_1,
                  "hessian_parallel_column_2" : column_2,
                },
                loop = {"file_name" : file_name, "val_ind" : val_ind, "freeze_ind" : freeze_ind, "column_1" : column_1, "column_2" : column_2},
              )


  # Get the Hessian matrix running in parallel
  if args.step == "HessianCollect":
    print(f"<< Collecting the Hessian matrix >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
              "method" : "HessianCollect",
              "hessian_input" : f"{data_dir}/HessianParallel{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
              "data_output" : f"{data_dir}/Hessian{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
              "extra_file_name" : str(val_ind),
              "freeze" : freeze["freeze"],
              "val_ind" : val_ind,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "freeze_ind" : freeze_ind},
          )


  # Get the Hessian matrix numerically
  if (args.step == "HessianNumerical") or ((args.step == "Hessian") and (args.likelihood_type in ["binned", "binned_extended"])):
    print(f"<< Calculating the Hessian matrix numerically >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
              "method" : "HessianNumerical",
              "best_fit_input" : f"{data_dir}/InitialFit{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
              "data_output" : f"{data_dir}/Hessian{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
              "extra_file_name" : str(val_ind),
              "freeze" : freeze["freeze"],
              "val_ind" : val_ind,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "freeze_ind" : freeze_ind},
          )


  # Get the Covariance matrix
  if args.step == "Covariance":
    print(f"<< Calculating the Covariance matrix >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
              "method" : "Covariance",
              "hessian_input" : f"{data_dir}/Hessian{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
              "data_output" : f"{data_dir}/Covariance{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
              "extra_file_name" : str(val_ind),
              "freeze" : freeze["freeze"],
              "val_ind" : val_ind,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "freeze_ind" : freeze_ind},
          )


  # Get the D matrix
  if args.step == "DMatrix":
    print(f"<< Calculating the D matrix >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
              "method" : "DMatrix",
              "best_fit_input" : f"{data_dir}/InitialFit{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
              "data_output" : f"{data_dir}/DMatrix{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
              "extra_file_name" : str(val_ind),
              "freeze" : freeze["freeze"],
              "val_ind" : val_ind,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "freeze_ind" : freeze_ind},
          )


  # Get the Hessian, D matrix and the Covariance matrix
  if args.step == "CovarianceWithDMatrix":
    print(f"<< Calculating the Covariance matrix with the D matrix correction >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
              "method" : "CovarianceWithDMatrix",
              "hessian_input" : f"{data_dir}/Hessian{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
              "d_matrix_input" : f"{data_dir}/DMatrix{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
              "data_output" : f"{data_dir}/CovarianceWithDMatrix{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
              "extra_file_name" : str(val_ind),
              "freeze" : freeze["freeze"],
              "val_ind" : val_ind,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "freeze_ind" : freeze_ind},
          )


  # Find sensible scan points
  if args.step in ["ScanPointsFromApproximate","ScanPointsFromHessian"]:
    print(f"<< Finding points to scan over >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for column in GetParameterLoop(file_name, cfg, include_nuisances=args.loop_over_nuisances, include_rate=args.loop_over_rates, include_lnN=args.loop_over_lnN):
          for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
            module.Run(
              module_name = "infer",
              class_name = "Infer",
              config = {
                **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
                "method" : args.step,
                "best_fit_input" : f"{data_dir}/InitialFit{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
                "hessian_input" : f"{data_dir}/Hessian{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
                "data_output" : f"{data_dir}/ScanPoints{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
                "extra_file_name" : str(val_ind),
                "freeze" : freeze["freeze"],
                "val_ind" : val_ind,
                "column" : column,
                "sigma_between_scan_points" : args.sigma_between_scan_points,
                "number_of_scan_points" : args.number_of_scan_points,

              },
              loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column, "freeze_ind" : freeze_ind},
            )


  # Run profiled likelihood scan
  if args.step == "Scan":
    print(f"<< Running profiled likelihood scans >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for column in GetParameterLoop(file_name, cfg, include_nuisances=args.loop_over_nuisances, include_rate=args.loop_over_rates, include_lnN=args.loop_over_lnN):
          for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
            for scan_ind in range(args.number_of_scan_points):
              module.Run(
                module_name = "infer",
                class_name = "Infer",
                config = {
                  **CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind),
                  "method" : args.step,
                  "best_fit_input" : f"{data_dir}/InitialFit{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
                  "hessian_input" : f"{data_dir}/Hessian{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
                  "data_output" : f"{data_dir}/Scan{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
                  "extra_file_name" : str(val_ind),
                  "freeze" : freeze["freeze"],
                  "val_ind" : val_ind,
                  "column" : column,
                  "sigma_between_scan_points" : args.sigma_between_scan_points,
                  "number_of_scan_points" : args.number_of_scan_points,
                  "scan_value" : GetDictionaryEntryFromYaml(f"{data_dir}/ScanPoints{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}/scan_ranges_{column}_{val_ind}.yaml", ["scan_values",scan_ind]),
                  "scan_ind" : str(scan_ind),
                  "other_input_files": [f"{data_dir}/ScanPoints{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}/scan_ranges_{column}_{val_ind}.yaml"],
                },
                loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column, "freeze_ind" : freeze_ind, "scan_ind" : scan_ind},
                save_class = not ((scan_ind + 1 == args.number_of_scan_points))
              )


  # Collect likelihood scan
  if args.step == "ScanCollect":
    print(f"<< Collecting likelihood scan results >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for column in GetParameterLoop(file_name, cfg, include_nuisances=args.loop_over_nuisances, include_rate=args.loop_over_rates, include_lnN=args.loop_over_lnN):
          for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
            module.Run(
              module_name = "scan_collect",
              class_name = "ScanCollect",
              config = {
                "number_of_scan_points" : args.number_of_scan_points,
                "column" : column,
                "data_input" : f"{data_dir}/Scan{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
                "data_output" : f"{data_dir}/ScanCollect{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}",
                "extra_file_name" : str(val_ind),
                "verbose" : not args.quiet,
              },
              loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column, "freeze_ind" : freeze_ind},
            )          


  # Plot likelihood scan
  if args.step == "ScanPlot":
    print(f"<< Plot likelihood scan >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for column in GetParameterLoop(file_name, cfg, include_nuisances=args.loop_over_nuisances, include_rate=args.loop_over_rates, include_lnN=args.loop_over_lnN):
          for freeze_ind, freeze in enumerate(GetFreezeLoop(args.freeze, val_info, file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, loop_over_nuisances=args.loop_over_nuisances, loop_over_rates=args.loop_over_rates, loop_over_lnN=args.loop_over_lnN)):
            module.Run(
              module_name = "scan_plot",
              class_name = "ScanPlot",
              config = {
                "column" : column,
                "data_input" : f"{data_dir}/ScanCollect{args.extra_input_dir_name}{freeze['extra_name']}/{file_name}",
                "plots_output" : f"{plots_dir}/ScanPlot{args.extra_output_dir_name}{freeze['extra_name']}/{file_name}", 
                "extra_file_name" : str(val_ind),
                "other_input" : {other_input.split(':')[0] : f"{data_dir}/{file_name}/{other_input.split(':')[1]}" for other_input in args.other_input.split(",")} if args.other_input is not None else {},
                "extra_plot_name" : args.extra_plot_name,
                "val_info" : val_info,
                "verbose" : not args.quiet,
              },
              loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column, "freeze_ind" : freeze_ind},
            ) 


  # Make the postfit asimov datasets
  if args.step == "MakePostFitAsimov":
    print(f"<< Making the postfit asimov datasets >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      asimov_file_names = [file_name] if file_name != "combined" else GetModelFileLoop(cfg)
      for asimov_file_name in asimov_file_names:
        for val_ind, val_info in enumerate(GetValidationLoop(cfg, asimov_file_name)):
          if SkipNonDensity(cfg, asimov_file_name, val_info, skip_non_density=args.skip_non_density): continue
          if SkipNonData(cfg, file_name, args.data_type, val_ind, allow_split=True): continue
          best_fit = GetBestFitFromYaml(f"{data_dir}/InitialFit{args.extra_input_dir_name}/{file_name}/best_fit_{val_ind}.yaml", cfg, file_name, prefit_nuisance_values=args.prefit_nuisance_values)
          for category in GetCategoryLoop(cfg):
            module.Run(
              module_name = "make_asimov",
              class_name = "MakeAsimov",
              config = {
                "cfg" : args.cfg,
                "density_model" : GetModelLoop(cfg, model_file_name=asimov_file_name, only_density=True, specific_category=category)[0],
                "regression_models" : GetModelLoop(cfg, model_file_name=asimov_file_name, only_regression=True, specific_category=category),
                "regression_spline_input" : f"{data_dir}/EvaluateRegression",
                "model_input" : f"{models_dir}",
                "extra_density_model_name" : args.extra_density_model_name,
                "extra_regression_model_name" : args.extra_regression_model_name,
                "parameters" : f"{data_dir}/PreProcess/{asimov_file_name}/{category}/parameters.yaml",
                "data_output" : f"{data_dir}/MakePostFitAsimov{args.extra_output_dir_name}/{file_name}/{asimov_file_name}/{category}/val_ind_{val_ind}",
                "n_asimov_events" : args.number_of_asimov_events,
                "seed" : args.asimov_seed,
                "val_info" : best_fit,
                "val_ind" : val_ind,
                "only_density" : args.only_density,
                "file_name" : asimov_file_name,
                "use_asimov_scaling" : args.use_asimov_scaling,
                "verbose" : not args.quiet,
              },
              loop = {"file_name" : file_name, "asimov_file_name": asimov_file_name, "val_ind" : val_ind, "category" : category},
            )


  # Make asimov datasets for uncertainty bands for postfit plots
  if args.step == "MakePostFitUncertaintyAsimov":
    print(f"<< Making the postfit asimov datasets for the uncertainty bands >>")
    for file_name in GetModelFileLoop(cfg):
      asimov_file_names = [file_name] if file_name != "combined" else GetModelFileLoop(cfg)
      for asimov_file_name in asimov_file_names:
        for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
          if SkipNonDensity(cfg, asimov_file_name, val_info, skip_non_density=args.skip_non_density): continue
          if SkipNonData(cfg, file_name, args.data_type, val_ind, allow_split=True): continue
          for category in GetCategoryLoop(cfg):
            for nuisance in GetParametersInModel(file_name, cfg, include_lnN=True, only_nuisances=True):
              for nuisance_value in ["up","down"]:
                summary_from = args.summary_from if args.summary_from not in ["Scan","Bootstrap"] else args.summary_from+"Collect"
                val_info = GetBestFitWithShiftedNuisancesFromYaml(f"{data_dir}/InitialFit{args.extra_input_dir_name}/{file_name}/best_fit_{val_ind}.yaml", f"{data_dir}/{args.summary_from}{args.extra_input_dir_name}/{bf_file_name}/{summary_from.lower()}_results_{nuisance}_{val_ind}.yaml", cfg, file_name, nuisance_value, prefit_nuisance_values=args.prefit_nuisance_values)
                module.Run(
                  module_name = "make_asimov",
                  class_name = "MakeAsimov",
                  config = {
                    "cfg" : args.cfg,
                    "density_model" : GetModelLoop(cfg, model_file_name=asimov_file_name, only_density=True, specific_category=category)[0],
                    "regression_models" : GetModelLoop(cfg, model_file_name=asimov_file_name, only_regression=True, specific_category=category),
                    "regression_spline_input" : f"{data_dir}/EvaluateRegression",
                    "model_input" : f"{models_dir}",
                    "extra_density_model_name" : args.extra_density_model_name,
                    "extra_regression_model_name" : args.extra_regression_model_name,
                    "parameters" : f"{data_dir}/PreProcess/{asimov_file_name}/{category}/parameters.yaml",
                    "data_output" : f"{data_dir}/MakePostFitUncertaintyAsimov{args.extra_output_dir_name}/{file_name}/{asimov_file_name}/{category}/val_ind_{val_ind}/{nuisance}/{nuisance_value}",
                    "n_asimov_events" : args.number_of_asimov_events,
                    "seed" : args.asimov_seed,
                    "val_info" : val_info,
                    "val_ind" : val_ind,
                    "only_density" : args.only_density,
                    "file_name" : asimov_file_name,
                    "use_asimov_scaling" : args.use_asimov_scaling,
                    "verbose" : not args.quiet,
                  },
                  loop = {"file_name" : file_name, "asimov_file_name" : asimov_file_name, "val_ind" : val_ind, "category" : category, "nuisance" : nuisance, "nuisance_value" : nuisance_value},
                )
  

  # Making plots using the network as a generator for individual Y values
  if args.step == "PostFitPlot":
    print(f"<< Drawing the distributions for the best fit values >>")
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        if SkipNonDensity(cfg, file_name, val_info, skip_non_density=args.skip_non_density): continue
        if SkipNonData(cfg, file_name, args.data_type, val_ind): continue
        for category in GetCategoryLoop(cfg):
          if args.likelihood_type in ["unbinned", "unbinned_extended"]:
            module.Run(
              module_name = "generator",
              class_name = "Generator",
              config = {
                "cfg" : args.cfg,
                "data_input" : GetDataInput(args.data_type, cfg, file_name, val_ind, data_dir, sim_type=args.sim_type, asimov_dir_name=f"MakeAsimov{args.extra_input_dir_name}")[category],
                "asimov_input": GetDataInput("asimov", cfg, file_name, val_ind, data_dir, sim_type=args.sim_type, asimov_dir_name=f"MakePostFitAsimov{args.extra_input_dir_name}/{file_name}")[category],
                "plots_output" : f"{plots_dir}/PostFitPlot{args.extra_output_dir_name}/{file_name}/{category}",
                "do_2d_unrolled" : args.plot_2d_unrolled,
                "extra_plot_name" : f"{val_ind}_{args.extra_plot_name}" if args.extra_plot_name != "" else str(val_ind),
                "sim_type" : args.sim_type,
                "val_info" : {},
                "plot_styles" : [1],
                "data_label" : {"Data":"data", "Simulated":"sim", "Asimov":"asimov"}[args.data_type],
                "stack_label" : "",
                "include_postfit_uncertainty" : args.include_postfit_uncertainty,
                "uncertainty_input" : {fn : {nuisance : {nuisance_value : f"{data_dir}/MakePostFitUncertaintyAsimov{args.extra_input_dir_name}/{fn}/{category}/val_ind_{val_ind}/{nuisance}/{nuisance_value}/asimov.parquet" for nuisance_value in ["up","down"]} for nuisance in GetParametersInModel(fn, cfg, include_lnN=True, only_nuisances=True)} for fn in (GetModelFileLoop(cfg) if file_name=="combined" else [file_name])},
                "use_expected_data_uncertainty" : args.use_expected_data_uncertainty,
                "verbose" : not args.quiet,
              },
              loop = {"file_name" : file_name, "val_ind" : val_ind, "category" : category},
            )
          else:
            raise NotImplementedError("PostFitPlot is not implemented for binned likelihoods yet.")


  # Calculate the chi squared of the summary
  if args.step == "SummaryChiSquared":
    print(f"<< Getting the chi squared of the summary >>")
    summary_from = args.summary_from if args.summary_from not in ["Scan","Bootstrap"] else args.summary_from+"Collect"
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      column_loop = GetParameterLoop(file_name, cfg, include_nuisances=args.loop_over_nuisances, include_rate=args.loop_over_rates, include_lnN=args.loop_over_lnN, include_per_model_rate=args.include_per_model_rate, include_per_model_lnN=args.include_per_model_lnN)
      if len(column_loop) == 0: continue
      validation_loop = GetValidationLoop(cfg, file_name)
      module.Run(
        module_name = "summary_chi_squared",
        class_name = "SummaryChiSquared",
        config = {
          "val_loop" : validation_loop,
          "data_input" : f"{data_dir}/{summary_from}{args.extra_input_dir_name}/{file_name}",
          "data_output" : f"{data_dir}/SummaryChiSquared{summary_from}{args.extra_output_dir_name}/{file_name}",
          "file_name" : f"{summary_from}_results".lower(),
          "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None else {},
          "column_loop" : column_loop,
          "verbose" : not args.quiet,
        },
        loop = {"file_name" : file_name},
      )


  # Collect all-but-one results for summary
  if args.step == "SummaryAllButOneCollect":
    print(f"<< Collecting all-but-one results for the summary plot >>")
    summary_from = args.summary_from if args.summary_from not in ["Scan","Bootstrap"] else args.summary_from+"Collect"
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, _ in enumerate(GetValidationLoop(cfg, file_name)):
        column_loop = GetParameterLoop(file_name, cfg, include_nuisances=args.loop_over_nuisances, include_rate=args.loop_over_rates, include_lnN=args.loop_over_lnN, include_per_model_rate=args.include_per_model_rate, include_per_model_lnN=args.include_per_model_lnN)
        if len(column_loop) <= 1: continue
        module.Run(
          module_name = "summary_all_but_one_collect",
          class_name = "SummaryAllButOneCollect",
          config = {
            "file_name" : file_name,
            "val_ind" : val_ind,
            "data_input" : f"{data_dir}/{args.summary_from}{args.extra_input_dir_name}",
            "data_output" : f"{data_dir}/{args.summary_from}{args.extra_output_dir_name}/{file_name}",
            "verbose" : not args.quiet,
            "summary_from" : summary_from,
            "column_loop" : column_loop,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )


  # Plot the summary of the results
  if args.step == "Summary":
    print(f"<< Plot the summary of results >>")
    summary_from = args.summary_from if args.summary_from not in ["Scan","Bootstrap"] else args.summary_from+"Collect"
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      column_loop = GetParameterLoop(file_name, cfg, include_nuisances=args.loop_over_nuisances, include_rate=args.loop_over_rates, include_lnN=args.loop_over_lnN, include_per_model_rate=args.include_per_model_rate, include_per_model_lnN=args.include_per_model_lnN)
      if len(column_loop) == 0: continue
      validation_loop = GetValidationLoop(cfg, file_name)
      module.Run(
        module_name = "summary",
        class_name = "Summary",
        config = {
          "val_loop" : validation_loop,
          "data_input" : f"{data_dir}/{summary_from}{args.extra_input_dir_name}/{file_name}",
          "plots_output" : f"{plots_dir}/Summary{args.summary_from}Plot{args.extra_output_dir_name}/{file_name}",
          "file_name" : f"{args.summary_from}_results".lower(),
          "other_input" : {other_input.split(':')[0] : [f"{data_dir}/{other_input.split(':')[1]}/{file_name}", other_input.split(':')[2]] for other_input in args.other_input.split(",")} if args.other_input is not None else {},
          "extra_plot_name" : args.extra_plot_name,
          "show2sigma" : args.summary_show_2sigma,
          "chi_squared" : None if not args.summary_show_chi_squared else GetDictionaryEntryFromYaml(f"{data_dir}/SummaryChiSquared{args.summary_from}{args.extra_input_dir_name}/{file_name}/summary_chi_squared.yaml", []),
          "nominal_name" : args.summary_nominal_name,
          "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None and args.freeze != "all-but-one" else {},
          "column_loop" : column_loop,
          "subtract" : args.summary_subtract,
          "verbose" : not args.quiet,
        },
        loop = {"file_name" : file_name},
      )


  # Plot the summary of the results per validation index
  if args.step == "SummaryPerVal":
    print(f"<< Plot the summary of results per validation index >>")
    summary_from = args.summary_from if args.summary_from not in ["Scan","Bootstrap"] else args.summary_from+"Collect"
    for file_name in GetModelFileLoop(cfg, with_combined=True):
      for val_ind, val_info in enumerate(GetValidationLoop(cfg, file_name)):
        column_loop = GetParameterLoop(file_name, cfg, include_nuisances=True, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN, include_per_model_rate=args.include_per_model_rate, include_per_model_lnN=args.include_per_model_lnN)
        if len(column_loop) == 0: continue
        module.Run(
          module_name = "summary_per_val",
          class_name = "SummaryPerVal",
          config = {
            "val_info" : val_info,
            "val_ind" : val_ind,
            "data_input" : f"{data_dir}/{summary_from}{args.extra_input_dir_name}/{file_name}",
            "plots_output" : f"{plots_dir}/SummaryPerVal{args.summary_from}Plot{args.extra_output_dir_name}/{file_name}",
            "file_name" : f"{args.summary_from}_results".lower(),
            "other_input" : {other_input.split(':')[0] : [f"{data_dir}/{other_input.split(':')[1]}/{file_name}", other_input.split(':')[2]] for other_input in args.other_input.split(",")} if args.other_input is not None else {},
            "extra_plot_name" : args.extra_plot_name,
            "show2sigma" : args.summary_show_2sigma,
            "nominal_name" : args.summary_nominal_name,
            "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None and args.freeze != "all-but-one" else {},
            "column_loop" : column_loop,
            "verbose" : not args.quiet,
            "constraints" : cfg["inference"]["nuisance_constraints"],
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )


  # Run the sweep
  module.Sweep()


if __name__ == "__main__":

  # Start the timer
  start_time = time.time()

  # Print title
  title = pyg.figlet_format("INNFER")
  print()
  print(title)

  # Parse the arguments
  args, default_args = parse_args()

  if args.step != "SnakeMake": # Run a non snakemake step

    # Loop through steps
    steps = args.step.split(",")
    for step in steps:
      args.step = step
      main(args, default_args)

  else: # Run snakemake

    # Setting up snakemake file
    snakemake_file = SetupSnakeMakeFile(args, default_args, main)

    if not args.snakemake_dry_run: # Run snakemake
      os.system(f"snakemake --cores all --profile htcondor -s '{snakemake_file}' --unlock &> /dev/null")
      snakemake_extra = " --forceall" if args.snakemake_force else ""
      os.system(f"snakemake{snakemake_extra} --cores all --profile htcondor -s '{snakemake_file}'")

  # Print the time elapsed
  print("<< Finished running without error >>")
  end_time = time.time()
  hours, remainder = divmod(end_time-start_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f"<< Time elapsed: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds >>")