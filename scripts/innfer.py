import argparse
import copy
import os
import sys
import time
import yaml

import numpy as np
import pandas as pd
import pyfiglet as pyg

from module import Module
from useful_functions import (
    CommonInferConfigOptions,
    GetDictionaryEntryFromYaml,
    GetFileLoop,
    GetScanArchitectures,
    GetValidateInfo,
    SetupSnakeMakeFile,
    SplitValidationParameters
)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--architecture', help='Config for running', type=str, default='configs/architecture/default.yaml')
  parser.add_argument('--benchmark', help='Run from benchmark scenario', default=None)
  parser.add_argument('--binned-fit-input', help='The inputs to do a binned fit either just bins ("X1[0,50,100,200]" or categories and bins "(X2<100):X1[0,50,100,200];(X2>100):X1[0,50,200]")', default=None)
  parser.add_argument('--cfg', help='Config for running', default=None)
  parser.add_argument('--custom-module', help='Name of custom module', default=None)
  parser.add_argument('--data-type', help='The data type to use when running the Generator, Bootstrap or Infer step. Default is sim for Bootstrap, and asimov for Infer.', type=str, default='sim', choices=['data', 'asimov', 'sim'])
  parser.add_argument('--disable-tqdm', help='Disable tqdm when training.', action='store_true')
  parser.add_argument('--dry-run', help='Setup batch submission without running.', action='store_true')
  parser.add_argument('--extra-infer-dir-name', help='Add extra name to infer step data output directory', type=str, default='')
  parser.add_argument('--extra-infer-plot-name', help='Add extra name to infer step end of plot', type=str, default='')
  parser.add_argument('--freeze', help='Other inputs to likelihood and summary plotting', type=str, default=None)
  parser.add_argument('--hyperparameter-metric', help='Colon separated metric name and whether you want max or min, separated by a comma.', type=str, default='inference_chi_squared_test_inf:all,min')
  parser.add_argument('--likelihood-type', help='Type of likelihood to use for fitting.', type=str, default='unbinned_extended', choices=['unbinned_extended', 'unbinned', 'binned_extended', 'binned'])
  parser.add_argument('--make-snakemake-inputs', help='Make the snakemake input file', action='store_true')
  parser.add_argument('--minimisation-method', help='Method for minimisation', type=str, default='scipy')
  parser.add_argument('--model-type', help='Name of model type', type=str, default='BayesFlow')
  parser.add_argument('--number-of-asimov-events', help='The number of asimov events', type=int, default=10**6)
  parser.add_argument('--number-of-bootstraps', help='The number of bootstrap initial fits to run', type=int, default=100)
  parser.add_argument('--number-of-scan-points', help='The number of scan points run', type=int, default=41)
  parser.add_argument('--number-of-shuffles', help='The number of times to loop through the dataset when shuffling in preprocess', type=int, default=10)
  parser.add_argument('--number-of-trials', help='The number of trials to test for BayesianHyperparameterTuning', type=int, default=10)
  parser.add_argument('--no-constraint', help='Do not use the constraints', action='store_true')
  parser.add_argument('--other-input', help='Other inputs to likelihood and summary plotting', type=str, default=None)
  parser.add_argument('--plot-2d-unrolled', help='Make 2D unrolled plots when running generator.', action='store_true')
  parser.add_argument('--plot-transformed', help='Plot transformed variables when running generator.', action='store_true')
  parser.add_argument('--points-per-job', help='The number of points ran per job', type=int, default=1)
  parser.add_argument('--quiet', help='No verbose output.', action='store_true')
  parser.add_argument('--scale-to-eff-events', help='Scale to the number of effective events rather than the yield.', action='store_true')
  parser.add_argument('--scan-over-nuisances', help='Perform likelihood scans over nuisance parameters as well as POIs', action='store_true')
  parser.add_argument('--sigma-between-scan-points', help='The estimated unprofiled sigma between the scanning points', type=float, default=0.2)
  parser.add_argument('--snakemake-cfg', help='Config for running with snakemake', default=None)
  parser.add_argument('--snakemake-force', help='Force snakemake to execute all steps', action='store_true')
  parser.add_argument('--split-validation-files', help='Split the validation files.', action='store_true')
  parser.add_argument('--specific', help='Specific part of a step to run.', type=str, default='')
  parser.add_argument('--step', help='Step to run.', type=str, default=None)
  #parser.add_argument('--step', help='Step to run.', type=str, default=None, choices=['SnakeMake', 'MakeBenchmark', 'PreProcess', 'InputPlot', 'Train', 'PerformanceMetrics', 'HyperparameterScan', 'HyperparameterScanCollect', 'BayesianHyperparameterTuning', 'SplitValidationFiles', 'Flow', 'Generator', 'GeneratorSummary', 'BootstrapInitialFits', 'BootstrapCollect', 'BootstrapPlot', 'BootstrapSummary', 'MakeAsimov', 'InitialFit', 'ApproximateUncertainty', 'Hessian', 'DMatrix', 'Covariance', 'CovarianceWithDMatrix', 'ScanPoints', 'Scan', 'ScanCollect', 'ScanPlot', 'BestFitDistributions', 'SummaryChiSquared', 'Summary', 'LikelihoodDebug'])
  parser.add_argument('--submit', help='Batch to submit to', type=str, default=None)
  parser.add_argument('--summary-from', help='Summary from bootstrap or likelihood scan', type=str, default='Covariance', choices=['Scan', 'Bootstrap','ApproximateUncertainty','Covariance','CovarianceWithDMatrix'])
  parser.add_argument('--summary-nominal-name', help='Name of nominal summary points', type=str, default='Nominal')
  parser.add_argument('--summary-show-2sigma', help='Show 2 sigma band on the summary.', action='store_true')
  parser.add_argument('--summary-show-chi-squared', help='Add the chi squared value to the plot', action='store_true')
  parser.add_argument('--summary-subtract', help='Use subtraction instead of division in summary', action='store_true')
  parser.add_argument('--use-wandb', help='Use wandb for logging.', action='store_true')
  parser.add_argument('--wandb-project-name', help='Name of project on wandb', type=str, default='innfer')
  default_args = parser.parse_args([])
  args = parser.parse_args()

  # Check inputs
  if args.cfg is None and args.benchmark is None:
    raise ValueError("The --cfg or --benchmark is required.")
  if args.step is None:
    raise ValueError("The --step is required.")

  # Adjust input paths
  if os.path.exists(f"configs/run/{args.cfg}"): # Change cfg path
    args.cfg = f"configs/run/{args.cfg}"
  if os.path.exists(f"configs/architecture/{args.architecture}"): # Change architecture path
    args.architecture = f"configs/architecture/{args.architecture}"
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
    if ".yaml" not in args.benchmark:
      args.cfg = f"configs/run/Benchmark_{args.benchmark}.yaml"
    else:
      args.cfg = f"configs/run/Benchmark_{args.benchmark.split('/')[-1].split('.yaml')[0]}.yaml"
  if args.submit is not None:
    args.disable_tqdm = True

  # Adjust other inputs
  if args.model_type == "Benchmark":
    if ".yaml" not in args.benchmark:
      args.model_type = f"Benchmark_{args.benchmark}"
    else:
      with open(args.benchmark, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
      args.model_type = f"Benchmark_Dim1CfgToBenchmark_{cfg['name']}"

  return args, default_args


def main(args, default_args):

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
    with open(args.cfg, 'r') as yaml_file:
      cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    module.job_name = f"jobs/{cfg['name']}/innfer_{args.step}"

  # PreProcess the dataset
  if args.step == "PreProcess":
    print("<< Preprocessing datasets >>")
    for file_name, parquet_name in cfg["files"].items():
      split_nuisances = cfg["split_nuisance_models"] if "split_nuisance_models" in cfg.keys() else False
      for nuisance in cfg["nuisances"] if split_nuisances else [None]:
        module.Run(
          module_name = "preprocess",
          class_name = "PreProcess",
          config = {
            "cfg" : args.cfg,
            "file_name" : file_name,
            "data_output" : f"data/{cfg['name']}/{file_name}/PreProcess",
            "number_of_shuffles" : args.number_of_shuffles,
            "nuisance" : nuisance,
            "verbose" : not args.quiet,
          },
          loop = {"file_name" : file_name, "nuisance" : nuisance},
          force = True,
        )

  # Custom
  if args.step == "Custom":
    print("<< Custom module >>")
    module.Run(
      module_name = args.custom_module,
      class_name = args.custom_module,
      config = {
        "cfg" : args.cfg,
      },
      loop = {}
    )

  # Plot preprocess data
  if args.step == "InputPlot":
    print("<< Plotting datasets >>")
    for file_name in GetFileLoop(cfg):
      module.Run(
        module_name = "input_plot",
        class_name = "InputPlot",
        config = {
          "cfg" : args.cfg,
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "data_input" : f"data/{cfg['name']}/{file_name}/PreProcess",
          "plots_output" : f"plots/{cfg['name']}/{file_name}/InputPlot",
          "verbose" : not args.quiet,
        },
        loop = {"file_name" : file_name},
        force = True,
      )

  # Train network
  if args.step == "Train":
    print("<< Training the networks >>")
    for file_name in GetFileLoop(cfg):
      module.Run(
        module_name = "train",
        class_name = "Train",
        config = {
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "architecture" : args.architecture,
          "data_output" : f"models/{cfg['name']}/{file_name}",
          "plots_output" : f"plots/{cfg['name']}/{file_name}/Train/",
          "disable_tqdm" : args.disable_tqdm,
          "verbose" : not args.quiet,        
        },
        loop = {"file_name" : file_name}
      )

  # Get performance metrics
  if args.step == "PerformanceMetrics":
    print("<< Getting the performance metrics of the trained networks >>")
    for file_name in GetFileLoop(cfg):
      val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type="sim", skip_empty_Y=True)
      module.Run(
        module_name = "performance_metrics",
        class_name = "PerformanceMetrics",
        config = {
          "model" : f"models/{cfg['name']}/{file_name}/{file_name}.h5",
          "architecture" : f"models/{cfg['name']}/{file_name}/{file_name}_architecture.yaml",
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "data_output" : f"data/{cfg['name']}/{file_name}/PerformanceMetrics",
          "val_loop" : val_loop_info["val_loops"][file_name] if file_name in val_loop_info["val_loops"].keys() else {},
          "pois": cfg["pois"],
          "nuisances": cfg["nuisances"],
          "verbose" : not  args.quiet,        
        },
        loop = {"file_name" : file_name}
      )

  # Perform a hyperparameter scan
  if args.step == "HyperparameterScan":
    print("<< Running a hyperparameter scan >>")
    for file_name in GetFileLoop(cfg):
      for architecture_ind, architecture in enumerate(GetScanArchitectures(args.architecture, data_output=f"data/{cfg['name']}/{file_name}/HyperparameterScan/")):
        val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type="sim", skip_empty_Y=True)
        module.Run(
          module_name = "hyperparameter_scan",
          class_name = "HyperparameterScan",
          config = {
            "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
            "architecture" : architecture,
            "data_output" : f"data/{cfg['name']}/{file_name}/HyperparameterScan",
            "use_wandb" : args.use_wandb,
            "wandb_project_name" : args.wandb_project_name,
            "wandb_submit_name" : f"{cfg['name']}_{file_name}",
            "disable_tqdm" : args.disable_tqdm,
            "save_extra_name" : f"_{architecture_ind}",
            "val_loop" : val_loop_info["val_loops"][file_name] if file_name in val_loop_info["val_loops"].keys() else {},
            "pois": cfg["pois"],
            "nuisances": cfg["nuisances"],
            "verbose" : not args.quiet,        
          },
          loop = {"file_name" : file_name, "architecture_ind" : architecture_ind}
        )

  # Collect a hyperparameter scan
  if args.step == "HyperparameterScanCollect":
    print("<< Collecting hyperparameter scan >>")
    for file_name in GetFileLoop(cfg):
      module.Run(
        module_name = "hyperparameter_scan_collect",
        class_name = "HyperparameterScanCollect",
        config = {
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "save_extra_names" : [f"_{architecture_ind}" for architecture_ind in range(len(GetScanArchitectures(args.architecture, write=False)))],
          "data_input" : f"data/{cfg['name']}/{file_name}/HyperparameterScan",
          "data_output" : f"models/{cfg['name']}/{file_name}",
          "metric" : args.hyperparameter_metric,
          "verbose" : not args.quiet,        
        },
        loop = {"file_name" : file_name}
      )

  # Perform a hyperparameter scan
  if args.step == "BayesianHyperparameterTuning":
    print("<< Running a bayesian hyperparameter tuning >>")
    for file_name in GetFileLoop(cfg):
      val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type="sim", skip_empty_Y=True)
      module.Run(
        module_name = "bayesian_hyperparameter_tuning",
        class_name = "BayesianHyperparameterTuning",
        config = {
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "tune_architecture" : args.architecture,
          "data_output" : f"data/{cfg['name']}/{file_name}/BayesianHyperparameterTuning",
          "best_model_output" : f"models/{cfg['name']}/{file_name}",
          "use_wandb" : args.use_wandb,
          "wandb_project_name" : args.wandb_project_name,
          "wandb_submit_name" : f"{cfg['name']}_{file_name}",
          "disable_tqdm" : args.disable_tqdm,
          "verbose" : not args.quiet,
          "metric" : args.hyperparameter_metric,
          "n_trials" : args.number_of_trials,
          "val_loop" : val_loop_info["val_loops"][file_name] if file_name in val_loop_info["val_loops"].keys() else {},
          "pois": cfg["pois"],
          "nuisances": cfg["nuisances"],
        },
        loop = {"file_name" : file_name}
      )

  # Plot preprocess data
  if args.step == "SplitValidationFiles":
    print("<< Splitting the validation files >>")
    validate_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, skip_empty_Y=True)["val_loops"]
    for file_name in GetFileLoop(cfg):
      if file_name not in validate_info.keys(): continue
      module.Run(
        module_name = "split_validation_files",
        class_name = "SplitValidationFiles",
        config = {
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "data_output" : f"data/{cfg['name']}/{file_name}/SplitValidationFiles",
          "val_loop" : validate_info[file_name],
          "verbose" : not args.quiet,
        },
        loop = {"file_name" : file_name},
        force = True,
      )

  # Making plots using the network as a generator for individual Y values
  if args.step == "Flow":
    print("<< Making plots looking at the flow of the network >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      if file_name == "combined": continue
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "flow",
          class_name = "Flow",
          config = {
            "Y_sim" : val_info["row"],
            "parameters" : val_loop_info["parameters"][file_name] if not args.split_validation_files else SplitValidationParameters(val_loop_info["val_loops"], file_name, val_ind, cfg),
            "model" : val_loop_info["models"][file_name],
            "architecture" : val_loop_info["architectures"][file_name],
            "plots_output" : f"plots/{cfg['name']}/{file_name}/Flow{args.extra_infer_dir_name}",
            "extra_plot_name" : f"{val_ind}_{args.extra_infer_plot_name}" if args.extra_infer_plot_name != "" else str(val_ind),
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind}
        )

  # Making plots using the network as a generator for individual Y values
  if args.step == "Generator":
    print("<< Making plots using the network as a generator for individual Y values >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "generator",
          class_name = "Generator",
          config = {
            "Y_sim" : val_info["row"],
            "Y_synth" : val_info["row"],
            "parameters" : val_loop_info["parameters"][file_name] if not args.split_validation_files else  SplitValidationParameters(val_loop_info["val_loops"], file_name, val_ind, cfg),
            "model" : val_loop_info["models"][file_name],
            "architecture" : val_loop_info["architectures"][file_name],
            "yield_function" : "default",
            "pois" : cfg["pois"],
            "nuisances" : cfg["nuisances"],
            "plots_output" : f"plots/{cfg['name']}/{file_name}/Generator{args.extra_infer_dir_name}",
            "scale_to_yield" : "extended" in args.likelihood_type,
            "do_2d_unrolled" : args.plot_2d_unrolled,
            "do_transformed" : args.plot_transformed,
            "extra_plot_name" : f"{val_ind}_{args.extra_infer_plot_name}" if args.extra_infer_plot_name != "" else str(val_ind),
            "data_type" : args.data_type,
            "verbose" : not args.quiet,
            "data_file" : cfg["data_file"],     
            "split_nuisance_models" : cfg["split_nuisance_models"] if "split_nuisance_models" in cfg.keys() and file_name == "combined" else False,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind}
        )

  # Making plots using the network as a generator for individual Y values
  if args.step == "GeneratorSummary":
    print("<< Making plots using the network as a generator summarising all Y values >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      module.Run(
        module_name = "generator_summary",
        class_name = "GeneratorSummary",
        config = {
          "val_loop" : val_loop,
          "parameters" : val_loop_info["parameters"][file_name],
          "model" : val_loop_info["models"][file_name],
          "architecture" : val_loop_info["architectures"][file_name],
          "yield_function" : "default",
          "pois" : cfg["pois"],
          "nuisances" : cfg["nuisances"],
          "plots_output" : f"plots/{cfg['name']}/{file_name}/GeneratorSummary{args.extra_infer_dir_name}",
          "scale_to_yield" : "extended" in args.likelihood_type,
          "extra_plot_name" : args.extra_infer_plot_name,
          "verbose" : not args.quiet,        
        },
        loop = {"file_name" : file_name},
      )

  # Make the asimov datasets
  if args.step == "MakeAsimov":
    print(f"<< Making the asimov datasets >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
            "method" : "MakeAsimov",
            "extra_file_name" : str(val_ind),
            "n_asimov_events" : args.number_of_asimov_events,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Run likelihood debug
  if args.step == "LikelihoodDebug":
    print(f"<< Running a single likelihood value for Y={args.other_input} >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
            "method" : "Debug",
            "extra_file_name" : str(val_ind),
            "other_input" : args.other_input,
            "model_type" : args.model_type,
            "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Run initial fits from a full dataset
  if args.step == "InitialFit":
    print(f"<< Running initial fits >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
            "method" : "InitialFit",
            "data_output" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
            "extra_file_name" : str(val_ind),
            "model_type" : args.model_type,
            "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Run approximate uncertainties
  if args.step == "ApproximateUncertainty":
    print(f"<< Finding the approximate uncertainties >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in [i for i in val_info["initial_best_fit_guess"].columns if i in (cfg["pois"] if not args.scan_over_nuisances else cfg["pois"]+cfg["nuisances"])]:
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
              "method" : "ApproximateUncertainty",
              "data_input" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
              "data_output" : f"data/{cfg['name']}/{file_name}/ApproximateUncertaintyCollect{args.extra_infer_dir_name}",
              "column" : column,
              "extra_file_name" : str(val_ind),
              "model_type" : args.model_type,
              "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
          )

  # Get the Hessian matrix
  if args.step == "Hessian":
    print(f"<< Calculating the Hessian matrix >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
            "method" : "Hessian",
            "data_input" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
            "data_output" : f"data/{cfg['name']}/{file_name}/Hessian{args.extra_infer_dir_name}",
            "model_type" : args.model_type,
            "extra_file_name" : str(val_ind),
            "data_file" : cfg["data_file"],
            "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
            "scan_over_nuisances" : args.scan_over_nuisances,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Get the Covariance matrix
  if args.step == "Covariance":
    print(f"<< Calculating the Covariance matrix >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
            "method" : "Covariance",
            "data_input" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
            "hessian_input" : f"data/{cfg['name']}/{file_name}/Hessian{args.extra_infer_dir_name}",
            "data_output" : f"data/{cfg['name']}/{file_name}/Covariance{args.extra_infer_dir_name}",
            "model_type" : args.model_type,
            "extra_file_name" : str(val_ind),
            "data_file" : cfg["data_file"],
            "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
            "scan_over_nuisances" : args.scan_over_nuisances,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Get the D matrix
  if args.step == "DMatrix":
    print(f"<< Calculating the D matrix >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
            "method" : "DMatrix",
            "data_input" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
            "data_output" : f"data/{cfg['name']}/{file_name}/DMatrix{args.extra_infer_dir_name}",
            "model_type" : args.model_type,
            "extra_file_name" : str(val_ind),
            "data_file" : cfg["data_file"],
            "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
            "scan_over_nuisances" : args.scan_over_nuisances,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Get the Hessian, D matrix and the Covariance matrix
  if args.step == "CovarianceWithDMatrix":
    print(f"<< Calculating the Covariance matrix with the D matrix correction >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
            "method" : "CovarianceWithDMatrix",
            "data_input" : f"data/{cfg['name']}/{file_name}/DMatrix{args.extra_infer_dir_name}",
            "hessian_input" : f"data/{cfg['name']}/{file_name}/Hessian{args.extra_infer_dir_name}",
            "data_output" : f"data/{cfg['name']}/{file_name}/CovarianceWithDMatrix{args.extra_infer_dir_name}",
            "model_type" : args.model_type,
            "extra_file_name" : str(val_ind),
            "data_file" : cfg["data_file"],
            "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
            "scan_over_nuisances" : args.scan_over_nuisances,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Find sensible scan points
  if args.step == "ScanPoints":
    print(f"<< Finding points to scan over >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in [i for i in val_info["initial_best_fit_guess"].columns if i in (cfg["pois"] if not args.scan_over_nuisances else cfg["pois"]+cfg["nuisances"])]:
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
              "method" : "ScanPoints",
              "data_input" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
              "data_output" : f"data/{cfg['name']}/{file_name}/ScanPoints{args.extra_infer_dir_name}",
              "column" : column,
              "sigma_between_scan_points" : args.sigma_between_scan_points,
              "number_of_scan_points" : args.number_of_scan_points,
              "extra_file_name" : str(val_ind),
              "model_type" : args.model_type,
              "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
          )

  # Run profiled likelihood scan
  if args.step == "Scan":
    print(f"<< Running profiled likelihood scans >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in [i for i in val_info["initial_best_fit_guess"].columns if i in (cfg["pois"] if not args.scan_over_nuisances else cfg["pois"]+cfg["nuisances"])]:
          for scan_ind in range(args.number_of_scan_points):
            module.Run(
              module_name = "infer",
              class_name = "Infer",
              config = {
                **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
                "method" : "Scan",
                "data_input" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
                "data_output" : f"data/{cfg['name']}/{file_name}/Scan{args.extra_infer_dir_name}",
                "column" : column,
                "scan_value" : GetDictionaryEntryFromYaml(f"data/{cfg['name']}/{file_name}/ScanPoints{args.extra_infer_dir_name}/scan_ranges_{column}_{val_ind}.yaml", ["scan_values",scan_ind]),
                "scan_ind" : str(scan_ind),
                "extra_file_name" : str(val_ind),
                "other_input_files": [f"data/{cfg['name']}/{file_name}/ScanPoints{args.extra_infer_dir_name}/scan_ranges_{column}_{val_ind}.yaml"],
                "model_type" : args.model_type,
                "asimov_input" : f"data/{cfg['name']}/{file_name}/MakeAsimov{args.extra_infer_dir_name}",
              },
              loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column, "scan_ind" : scan_ind},
              save_class = not ((scan_ind + 1 == args.number_of_scan_points))
            )

  # Collect likelihood scan
  if args.step == "ScanCollect":
    print(f"<< Collecting likelihood scan results >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in [i for i in val_info["initial_best_fit_guess"].columns if i in (cfg["pois"] if not args.scan_over_nuisances else cfg["pois"]+cfg["nuisances"])]:
          module.Run(
            module_name = "scan_collect",
            class_name = "ScanCollect",
            config = {
              "number_of_scan_points" : args.number_of_scan_points,
              "column" : column,
              "data_input" : f"data/{cfg['name']}/{file_name}/Scan{args.extra_infer_dir_name}",
              "data_output" : f"data/{cfg['name']}/{file_name}/ScanCollect{args.extra_infer_dir_name}", 
              "extra_file_name" : str(val_ind),
              "verbose" : not args.quiet,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
          )


  # Plot likelihood scan
  if args.step == "ScanPlot":
    print(f"<< Plot likelihood scan >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in [i for i in val_info["initial_best_fit_guess"].columns if i in (cfg["pois"] if not args.scan_over_nuisances else cfg["pois"]+cfg["nuisances"])]:
          module.Run(
            module_name = "scan_plot",
            class_name = "ScanPlot",
            config = {
              "column" : column,
              "data_input" : f"data/{cfg['name']}/{file_name}/ScanCollect{args.extra_infer_dir_name}",
              "plots_output" : f"plots/{cfg['name']}/{file_name}/ScanPlot{args.extra_infer_dir_name}", 
              "extra_file_name" : str(val_ind),
              "other_input" : {other_input.split(':')[0] : f"data/{cfg['name']}/{file_name}/{other_input.split(':')[1]}" for other_input in args.other_input.split(",")} if args.other_input is not None else {},
              "extra_plot_name" : args.extra_infer_plot_name,
              "verbose" : not args.quiet,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
          ) 

  # Bootstrap initial fits
  if args.step == "BootstrapInitialFits":
    print(f"<< Bootstrapping the initial fits >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for bootstrap_ind in range(args.number_of_bootstraps):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              **CommonInferConfigOptions(args, cfg, val_info, val_loop_info, file_name, val_ind),
              "method" : "InitialFit",
              "data_output" : f"data/{cfg['name']}/{file_name}/BootstrapInitialFits{args.extra_infer_dir_name}",
              "plots_output" : f"plots/{cfg['name']}/{file_name}/BootstrapInitialFits{args.extra_infer_dir_name}",
              "resample" : True,
              "resampling_seed" : bootstrap_ind,
              "extra_file_name" : f"{val_ind}_{bootstrap_ind}",
              "model_type" : args.model_type,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "bootstrap_ind": bootstrap_ind},
        )

  # Collect boostrapped fits
  if args.step == "BootstrapCollect":
    print(f"<< Collecting the initial fits >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "bootstrap_collect",
          class_name = "BootstrapCollect",
          config = {
            "number_of_bootstraps" : args.number_of_bootstraps,
            "columns" : list(val_info["initial_best_fit_guess"].columns),
            "data_input" : f"data/{cfg['name']}/{file_name}/BootstrapInitialFits{args.extra_infer_dir_name}",
            "data_output" : f"data/{cfg['name']}/{file_name}/BootstrapCollect{args.extra_infer_dir_name}",
            "extra_file_name" : f"{val_ind}",
            "verbose" : not args.quiet,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Plot the boostrapped fits
  if args.step == "BootstrapPlot":
    print(f"<< Plot the bootstrapped fits >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in list(val_info["initial_best_fit_guess"].columns):
          module.Run(
            module_name = "bootstrap_plot",
            class_name = "BootstrapPlot",
            config = {
              "column" : column,
              "data_input" : f"data/{cfg['name']}/{file_name}/BootstrapCollect{args.extra_infer_dir_name}",
              "plots_output" : f"plots/{cfg['name']}/{file_name}/BootstrapPlot{args.extra_infer_dir_name}",
              "extra_file_name" : f"{val_ind}",
              "extra_plot_name" : args.extra_infer_plot_name,
              "verbose" : not args.quiet,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
        )

  # Draw best fit distributions
  if args.step == "BestFitDistributions":
    print(f"<< Drawing the distributions for the best fit values >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        best_fit = GetDictionaryEntryFromYaml(f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}/best_fit_{val_ind}.yaml", [])
        if args.likelihood_type in ["unbinned", "unbinned_extended"]:
          module.Run(
            module_name = "generator",
            class_name = "Generator",
            config = {
              "Y_sim" : val_info["row"],
              "Y_synth" : pd.DataFrame([best_fit["best_fit"]], columns=best_fit["columns"], dtype=np.float64) if best_fit is not None else None,
              "data_type" : args.data_type if args.data_type is not None else "sim",
              "parameters" : val_loop_info["parameters"][file_name],
              "model" : val_loop_info["models"][file_name],
              "architecture" : val_loop_info["architectures"][file_name],
              "yield_function" : "default",
              "pois" : cfg["pois"],
              "nuisances" : cfg["nuisances"],
              "plots_output" : f"plots/{cfg['name']}/{file_name}/BestFitDistributions{args.extra_infer_dir_name}",
              "scale_to_yield" : "extended" in args.likelihood_type,
              "do_2d_unrolled" : args.plot_2d_unrolled,
              "do_transformed" : args.plot_transformed,
              "extra_plot_name" : f"{val_ind}_{args.extra_infer_plot_name}" if args.extra_infer_plot_name != "" else str(val_ind),
              "other_input_files" : [f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}/best_fit_{val_ind}.yaml"],
              "verbose" : not args.quiet,
              "data_file" : cfg["data_file"],
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind}
          )
        elif args.likelihood_type in ["binned", "binned_extended"]:
          module.Run(
            module_name = "binned_distributions",
            class_name = "BinnedDistributions",
            config = {
              "Y_data" : val_info["row"],
              "Y_stack" : pd.DataFrame([best_fit["best_fit"]], columns=best_fit["columns"], dtype=np.float64) if best_fit is not None else None,
              "data_type" : args.data_type if args.data_type is not None else "sim",
              "binned_fit_input" : args.binned_fit_input,
              "parameters" : val_loop_info["parameters"][file_name],
              "pois" : cfg["pois"],
              "nuisances" : cfg["nuisances"],
              "plots_output" : f"plots/{cfg['name']}/{file_name}/BestFitDistributions{args.extra_infer_dir_name}",
              "scale_to_yield" : "extended" in args.likelihood_type,
              "extra_plot_name" : f"{val_ind}_{args.extra_infer_plot_name}" if args.extra_infer_plot_name != "" else str(val_ind),
              "other_input_files" : [f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}/best_fit_{val_ind}.yaml"],
              "verbose" : not args.quiet,
              "data_file" : cfg["data_file"],
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind}
          )

  # Calculate the chi squared of the summary
  if args.step == "SummaryChiSquared":
    print(f"<< Getting the chi squared of the summary >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    if args.summary_from in ["Scan","Bootstrap","ApproximateUncertainty"]: args.summary_from += "Collect"
    for file_name, val_loop in val_loop_info["val_loops"].items():
      module.Run(
        module_name = "summary_chi_squared",
        class_name = "SummaryChiSquared",
        config = {
          "val_loop" : val_loop,
          "data_input" : f"data/{cfg['name']}/{file_name}/{args.summary_from}{args.extra_infer_dir_name}",
          "data_output" : f"data/{cfg['name']}/{file_name}/SummaryChiSquared{args.summary_from}{args.extra_infer_dir_name}",
          "file_name" : f"{args.summary_from}_results".lower(),
          "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None else {},
          "column_loop" : [i for i in val_loop[0]["initial_best_fit_guess"].columns if i in (cfg["pois"] if not args.scan_over_nuisances else cfg["pois"]+cfg["nuisances"])],
          "verbose" : not args.quiet,
        },
        loop = {"file_name" : file_name},
      )

  # Plot the summary of the results
  if args.step == "Summary":
    print(f"<< Plot the summary of results >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=args.data_type, skip_empty_Y=True)
    summary_from = args.summary_from if args.summary_from not in ["Scan","Bootstrap","ApproximateUncertainty"] else args.summary_from+"Collect"
    for file_name, val_loop in val_loop_info["val_loops"].items():
      module.Run(
        module_name = "summary",
        class_name = "Summary",
        config = {
          "val_loop" : val_loop,
          "data_input" : f"data/{cfg['name']}/{file_name}/{summary_from}{args.extra_infer_dir_name}",
          "plots_output" : f"plots/{cfg['name']}/{file_name}/Summary{args.summary_from}Plot{args.extra_infer_dir_name}",
          "file_name" : f"{args.summary_from}_results".lower(),
          "other_input" : {other_input.split(':')[0] : [f"data/{cfg['name']}/{file_name}/{other_input.split(':')[1]}", other_input.split(':')[2]] for other_input in args.other_input.split(",")} if args.other_input is not None else {},
          "extra_plot_name" : args.extra_infer_plot_name,
          "show2sigma" : args.summary_show_2sigma,
          "chi_squared" : None if not args.summary_show_chi_squared else GetDictionaryEntryFromYaml(f"data/{cfg['name']}/{file_name}/SummaryChiSquared{args.summary_from}{args.extra_infer_dir_name}/summary_chi_squared.yaml", []),
          "nominal_name" : args.summary_nominal_name,
          "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None else {},
          "column_loop" : [i for i in val_loop[0]["initial_best_fit_guess"].columns if i in (cfg["pois"] if not args.scan_over_nuisances else cfg["pois"]+cfg["nuisances"])],
          "subtract" : args.summary_subtract,
          "verbose" : not args.quiet,
        },
        loop = {"file_name" : file_name},
      )

  module.Sweep()

if __name__ == "__main__":

  start_time = time.time()
  title = pyg.figlet_format("INNFER")
  print()
  print(title)

  args, default_args = parse_args()

  if args.step != "SnakeMake": # Run a non snakemake step

    # Loop through steps
    steps = args.step.split(",")
    for step in steps:
      args.step = step
      main(args, default_args)

  else:

    snakemake_file = SetupSnakeMakeFile(args, default_args, main)
    os.system(f"snakemake --cores all --profile htcondor -s '{snakemake_file}' --unlock &> /dev/null")
    snakemake_extra = " --forceall" if args.snakemake_force else ""
    os.system(f"snakemake{snakemake_extra} --cores all --profile htcondor -s '{snakemake_file}'")

  print("<< Finished running without error >>")
  end_time = time.time()
  hours, remainder = divmod(end_time-start_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f"<< Time elapsed: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds >>")