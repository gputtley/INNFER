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
    GetDictionaryEntryFromYaml,
    GetScanArchitectures,
    GetValidateInfo,
    SetupSnakeMakeFile
)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--architecture', help='Config for running', type=str, default='configs/architecture/default.yaml')
  parser.add_argument('--benchmark', help='Run from benchmark scenario', default=None)
  parser.add_argument('--cfg', help='Config for running', default=None)
  parser.add_argument('--data-type', help='The data type to use when running the Generator, Bootstrap or Infer step. Default is sim for Bootstrap, and asimov for Infer.', type=str, default=None, choices=['data', 'asimov', 'sim'])
  parser.add_argument('--disable-tqdm', help='Disable tqdm when training.', action='store_true')
  parser.add_argument('--dry-run', help='Setup batch submission without running.', action='store_true')
  parser.add_argument('--extra-infer-dir-name', help='Add extra name to infer step data output directory', type=str, default='')
  parser.add_argument('--extra-infer-plot-name', help='Add extra name to infer step end of plot', type=str, default='')
  parser.add_argument('--freeze', help='Other inputs to likelihood and summary plotting', type=str, default=None)
  parser.add_argument('--hyperparameter-scan-metric', help='Colon separated metric name and whether you want max or min, separated by a comma.', type=str, default='chi_squared_test:total,min')
  parser.add_argument('--likelihood-type', help='Type of likelihood to use for fitting.', type=str, default='unbinned_extended', choices=['unbinned_extended', 'unbinned', 'binned_extended', 'binned'])
  parser.add_argument('--make-snakemake-inputs', help='Make the snakemake input file', action='store_true')
  parser.add_argument('--model-type', help='Name of model type', type=str, default='BayesFlow')
  parser.add_argument('--number-of-bootstraps', help='The number of bootstrap initial fits to run', type=int, default=100)
  parser.add_argument('--number-of-scan-points', help='The number of scan points run', type=int, default=41)
  parser.add_argument('--other-input', help='Other inputs to likelihood and summary plotting', type=str, default=None)
  parser.add_argument('--plot-2d-unrolled', help='Make 2D unrolled plots when running generator.', action='store_true')
  parser.add_argument('--points-per-job', help='The number of points ran per job', type=int, default=1)
  parser.add_argument('--scale-to-eff-events', help='Scale to the number of effective events rather than the yield.', action='store_true')
  parser.add_argument('--sigma-between-scan-points', help='The estimated unprofiled sigma between the scanning points', type=float, default=0.2)
  parser.add_argument('--snakemake-cfg', help='Config for running with snakemake', default=None)
  parser.add_argument('--snakemake-force', help='Force snakemake to execute all steps', action='store_true')
  parser.add_argument('--specific', help='Specific part of a step to run.', type=str, default='')
  parser.add_argument('--step', help='Step to run.', type=str, default=None, choices=['SnakeMake', 'MakeBenchmark', 'PreProcess', 'Train', 'PerformanceMetrics', 'HyperparameterScan', 'HyperparameterScanCollect', 'Generator', 'GeneratorSummary', 'BootstrapInitialFits', 'BootstrapCollect', 'BootstrapPlot', 'BootstrapSummary', 'InitialFit', 'ScanPoints', 'Scan', 'ScanCollect', 'ScanPlot', 'BestFitDistributions', 'Summary', 'LikelihoodDebug'])
  parser.add_argument('--submit', help='Batch to submit to', type=str, default=None)
  parser.add_argument('--summary-from', help='Summary from bootstrap or likelihood scan', type=str, default='Scan', choices=['Scan', 'Bootstrap'])
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

  # Adjust other inputs
  if args.model_type == "Benchmark":
    args.model_type = f"Benchmark_{args.benchmark}"

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
      config = {"name" : args.benchmark},
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
      module.Run(
        module_name = "preprocess",
        class_name = "PreProcess",
        config = {
          "cfg" : args.cfg,
          "file_name" : file_name,
          "parquet_file_name" : parquet_name,
          "data_output" : f"data/{cfg['name']}/{file_name}/PreProcess",
          "plots_output" : f"plots/{cfg['name']}/{file_name}/PreProcess",
        },
        loop = {"file_name" : file_name},
        force = True,
      )

  # Train network
  if args.step == "Train":
    print("<< Training the networks >>")
    for file_name, parquet_name in cfg["files"].items():
      module.Run(
        module_name = "train",
        class_name = "Train",
        config = {
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "architecture" : args.architecture,
          "data_output" : f"models/{cfg['name']}/{file_name}",
          "plots_output" : f"plots/{cfg['name']}/{file_name}/Train/",
          "disable_tqdm" : args.disable_tqdm,
        },
        loop = {"file_name" : file_name}
      )

  # Get performance metrics
  if args.step == "PerformanceMetrics":
    print("<< Getting the performance metrics of the trained networks >>")
    for file_name, parquet_name in cfg["files"].items():
      module.Run(
        module_name = "performance_metrics",
        class_name = "PerformanceMetrics",
        config = {
          "model" : f"models/{cfg['name']}/{file_name}/{file_name}.h5",
          "architecture" : f"models/{cfg['name']}/{file_name}/{file_name}_architecture.yaml",
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "data_output" : f"data/{cfg['name']}/{file_name}/PerformanceMetrics",
        },
        loop = {"file_name" : file_name}
      )

  # Perform a hyperparameter scan
  if args.step == "HyperparameterScan":
    print("<< Running a hyperparameter scan >>")
    for file_name, parquet_name in cfg["files"].items():
      for architecture_ind, architecture in enumerate(GetScanArchitectures(args.architecture, data_output=f"data/{cfg['name']}/{file_name}/HyperparameterScan/")):
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
          },
          loop = {"file_name" : file_name, "architecture_ind" : architecture_ind}
        )

  # Collect a hyperparameter scan
  if args.step == "HyperparameterScanCollect":
    print("<< Collecting hyperparameter scan >>")
    for file_name, parquet_name in cfg["files"].items():
      module.Run(
        module_name = "hyperparameter_scan_collect",
        class_name = "HyperparameterScanCollect",
        config = {
          "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
          "save_extra_names" : [f"_{architecture_ind}" for architecture_ind in range(len(GetScanArchitectures(args.architecture, write=False)))],
          "data_input" : f"data/{cfg['name']}/{file_name}/HyperparameterScan",
          "data_output" : f"models/{cfg['name']}/{file_name}",
          "metric" : args.hyperparameter_scan_metric
        },
        loop = {"file_name" : file_name}
      )

  # Making plots using the network as a generator for individual Y values
  if args.step == "Generator":
    print("<< Making plots using the network as a generator for individual Y values >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}",cfg)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "generator",
          class_name = "Generator",
          config = {
            "Y_sim" : val_info["row"],
            "Y_synth" : val_info["row"],
            "parameters" : val_loop_info["parameters"][file_name],
            "model" : val_loop_info["models"][file_name],
            "architecture" : val_loop_info["architectures"][file_name],
            "yield_function" : "default",
            "pois" : cfg["pois"],
            "nuisances" : cfg["nuisances"],
            "plots_output" : f"plots/{cfg['name']}/{file_name}/Generator{args.extra_infer_dir_name}",
            "scale_to_yield" : "extended" in args.likelihood_type,
            "do_2d_unrolled" : args.plot_2d_unrolled,
            "extra_plot_name" : f"{val_ind}_{args.extra_infer_plot_name}" if args.extra_infer_plot_name != "" else str(val_ind),
            "data_type" : args.data_type if args.data_type is not None else "sim",
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
          "do_2d_unrolled" : args.plot_2d_unrolled,
          "extra_plot_name" : args.extra_infer_plot_name,
        },
        loop = {"file_name" : file_name},
      )

  # Run likelihood debug
  if args.step == "LikelihoodDebug":
    print(f"<< Running a single likelihood value for Y={args.other_input} for the {args.data_type} dataset >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            "method" : "Debug",
            "true_Y" : val_info["row"],
            "initial_best_fit_guess" : val_info["initial_best_fit_guess"],
            "data_type" : args.data_type if args.data_type is not None else "sim",
            "parameters" : val_loop_info["parameters"][file_name],
            "model" : val_loop_info["models"][file_name],
            "architecture" : val_loop_info["architectures"][file_name],
            "yield_function" : "default",
            "pois" : cfg["pois"],
            "nuisances" : cfg["nuisances"],
            "likelihood_type" : args.likelihood_type,
            "inference_options" : cfg["inference"] if file_name == "combined" else {},
            "scale_to_eff_events" : args.scale_to_eff_events,
            "other_input" : args.other_input,
            "model_type" : args.model_type,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Run initial fits from a full dataset
  if args.step == "InitialFit":
    print(f"<< Running initial fits for the {args.data_type} dataset >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        module.Run(
          module_name = "infer",
          class_name = "Infer",
          config = {
            "method" : "InitialFit",
            "true_Y" : val_info["row"],
            "initial_best_fit_guess" : val_info["initial_best_fit_guess"],
            "data_type" : args.data_type if args.data_type is not None else "asimov",
            "parameters" : val_loop_info["parameters"][file_name],
            "model" : val_loop_info["models"][file_name],
            "architecture" : val_loop_info["architectures"][file_name],
            "yield_function" : "default",
            "pois" : cfg["pois"],
            "nuisances" : cfg["nuisances"],
            "data_output" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
            "likelihood_type" : args.likelihood_type,
            "inference_options" : cfg["inference"] if file_name == "combined" else {},
            "resample" : False,
            "scale_to_eff_events" : args.scale_to_eff_events,
            "extra_file_name" : str(val_ind),
            "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None else {},
            "model_type" : args.model_type,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Find sensible scan points
  if args.step == "ScanPoints":
    print(f"<< Finding points to scan over >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in list(val_info["initial_best_fit_guess"].columns):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              "method" : "ScanPoints",
              "true_Y" : val_info["row"],
              "initial_best_fit_guess" : val_info["initial_best_fit_guess"],
              "data_type" : args.data_type if args.data_type is not None else "asimov",
              "parameters" : val_loop_info["parameters"][file_name],
              "model" : val_loop_info["models"][file_name],
              "architecture" : val_loop_info["architectures"][file_name],
              "yield_function" : "default",
              "pois" : cfg["pois"],
              "nuisances" : cfg["nuisances"],
              "data_input" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
              "data_output" : f"data/{cfg['name']}/{file_name}/ScanPoints{args.extra_infer_dir_name}",
              "likelihood_type" : args.likelihood_type,
              "inference_options" : cfg["inference"] if file_name == "combined" else {},
              "resample" : False,
              "scale_to_eff_events" : args.scale_to_eff_events,
              "column" : column,
              "sigma_between_scan_points" : args.sigma_between_scan_points,
              "number_of_scan_points" : args.number_of_scan_points,
              "extra_file_name" : str(val_ind),
              "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None else {},
              "model_type" : args.model_type,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
          )

  # Run profiled likelihood scan
  if args.step == "Scan":
    print(f"<< Running profiled likelihood scans >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in list(val_info["initial_best_fit_guess"].columns):
          for scan_ind in range(args.number_of_scan_points):
            module.Run(
              module_name = "infer",
              class_name = "Infer",
              config = {
                "method" : "Scan",
                "true_Y" : val_info["row"],
                "initial_best_fit_guess" : val_info["initial_best_fit_guess"],
                "data_type" : args.data_type if args.data_type is not None else "asimov",
                "parameters" : val_loop_info["parameters"][file_name],
                "model" : val_loop_info["models"][file_name],
                "architecture" : val_loop_info["architectures"][file_name],
                "yield_function" : "default",
                "pois" : cfg["pois"],
                "nuisances" : cfg["nuisances"],
                "data_input" : f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}",
                "data_output" : f"data/{cfg['name']}/{file_name}/Scan{args.extra_infer_dir_name}",
                "likelihood_type" : args.likelihood_type,
                "inference_options" : cfg["inference"] if file_name == "combined" else {},
                "resample" : False,
                "scale_to_eff_events" : args.scale_to_eff_events,
                "column" : column,
                "scan_value" : GetDictionaryEntryFromYaml(f"data/{cfg['name']}/{file_name}/ScanPoints{args.extra_infer_dir_name}/scan_ranges_{column}_{val_ind}.yaml", ["scan_values",scan_ind]),
                "scan_ind" : str(scan_ind),
                "extra_file_name" : str(val_ind),
                "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None else {},
                "other_input_files": [f"data/{cfg['name']}/{file_name}/ScanPoints{args.extra_infer_dir_name}/scan_ranges_{column}_{val_ind}.yaml"],
                "model_type" : args.model_type,
              },
              loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column, "scan_ind" : scan_ind},
            )

  # Collect likelihood scan
  if args.step == "ScanCollect":
    print(f"<< Collecting likelihood scan results >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in list(val_info["initial_best_fit_guess"].columns):
          module.Run(
            module_name = "scan_collect",
            class_name = "ScanCollect",
            config = {
              "number_of_scan_points" : args.number_of_scan_points,
              "column" : column,
              "data_input" : f"data/{cfg['name']}/{file_name}/Scan{args.extra_infer_dir_name}",
              "data_output" : f"data/{cfg['name']}/{file_name}/ScanCollect{args.extra_infer_dir_name}", 
              "extra_file_name" : str(val_ind),
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
          )


  # Plot likelihood scan
  if args.step == "ScanPlot":
    print(f"<< Plot likelihood scan >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for column in list(val_info["initial_best_fit_guess"].columns):
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
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
          )

  # Bootstrap initial fits
  if args.step == "BootstrapInitialFits":
    print(f"<< Bootstrapping the initial fits >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        for bootstrap_ind in range(args.number_of_bootstraps):
          module.Run(
            module_name = "infer",
            class_name = "Infer",
            config = {
              "method" : "InitialFit",
              "true_Y" : val_info["row"],
              "initial_best_fit_guess" : val_info["initial_best_fit_guess"],
              "data_type" : args.data_type if args.data_type is not None else "sim",
              "parameters" : val_loop_info["parameters"][file_name],
              "model" : val_loop_info["models"][file_name],
              "architecture" : val_loop_info["architectures"][file_name],
              "yield_function" : "default",
              "pois" : cfg["pois"],
              "nuisances" : cfg["nuisances"],
              "data_output" : f"data/{cfg['name']}/{file_name}/BootstrapInitialFits{args.extra_infer_dir_name}",
              "likelihood_type" : args.likelihood_type,
              "inference_options" : cfg["inference"] if file_name == "combined" else {},
              "resample" : True,
              "resampling_seed" : bootstrap_ind,
              "scale_to_eff_events" : args.scale_to_eff_events,
              "extra_file_name" : f"{val_ind}_{bootstrap_ind}",
              "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in args.freeze.split(",")} if args.freeze is not None else {},
              "model_type" : args.model_type,
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "bootstrap_ind": bootstrap_ind},
        )

  # Collect boostrapped fits
  if args.step == "BootstrapCollect":
    print(f"<< Collecting the initial fits >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
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
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind},
        )

  # Plot the boostrapped fits
  if args.step == "BootstrapPlot":
    print(f"<< Plot the bootstrapped fits >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
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
            },
            loop = {"file_name" : file_name, "val_ind" : val_ind, "column" : column},
        )

  # Draw best fit distributions
  if args.step == "BestFitDistributions":
    print(f"<< Drawing the distributions for the best values >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        best_fit = GetDictionaryEntryFromYaml(f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}/best_fit_{val_ind}.yaml", [])
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
            "extra_plot_name" : f"{val_ind}_{args.extra_infer_plot_name}" if args.extra_infer_plot_name != "" else str(val_ind),
            "other_input_files" : [f"data/{cfg['name']}/{file_name}/InitialFit{args.extra_infer_dir_name}/best_fit_{val_ind}.yaml"]
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind}
        )

  # Plot the summary of the results
  if args.step == "Summary":
    print(f"<< Plot the summary of results >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}", cfg, data_type=(args.data_type if args.data_type is not None else "asimov"), skip_empty_Y=True)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      module.Run(
        module_name = "summary",
        class_name = "Summary",
        config = {
          "columns" : list(val_loop[0]["initial_best_fit_guess"].columns),
          "val_loop" : val_loop,
          "data_input" : f"data/{cfg['name']}/{file_name}/{args.summary_from}Collect{args.extra_infer_dir_name}",
          "plots_output" : f"plots/{cfg['name']}/{file_name}/Summary{args.summary_from}Plot{args.extra_infer_dir_name}",
          "file_name" : f"{args.summary_from}_results".lower(),
          "other_input" : {other_input.split(':')[0] : [f"data/{cfg['name']}/{file_name}/{other_input.split(':')[1]}", other_input.split(':')[2]] for other_input in args.other_input.split(",")} if args.other_input is not None else {},
          "extra_plot_name" : args.extra_infer_plot_name,
        },
        loop = {"file_name" : file_name},
      )

  # Make PDFs of all plots
  if args.step == "MakePDFs":
    print(f"<< Making PDFs of all plots produced >>")

  module.Sweep()

if __name__ == "__main__":

  start_time = time.time()
  title = pyg.figlet_format("INNFER")
  print()
  print(title)

  args, default_args = parse_args()

  if args.step != "SnakeMake": # Run a non snakemake step

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