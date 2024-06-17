import argparse
import time
import os
import sys
import yaml
import copy
import pyfiglet as pyg

from useful_functions import GetScanArchitectures, GetValidateInfo, SetupSnakeMakeFile
from module import Module

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', help = 'Config for running', default = None)
  parser.add_argument('--snakemake-cfg', help = 'Config for running with snakemake', default = None)
  parser.add_argument('--benchmark', help = 'Run from benchmark scenario', default = None)
  parser.add_argument('--step', help = 'Step to run.', type = str, default = None, choices = ["SnakeMake","MakeBenchmark", "PreProcess", "Train",  "PerformanceMetrics", "HyperparameterScan", "HyperparameterScanCollect", "Generator", "GeneratorSummary", "BootstrapFits", "BootstrapCollect", "BootstrapPlot", "BootstrapSummary", "InferInitialFit", "InferScan", "InferCollect", "InferPlot", "InferSummary"])
  parser.add_argument('--specific', help = 'Specific part of a step to run.', type = str, default = "")
  parser.add_argument('--data-type', help = 'The data type to use when running the Generator, Bootstrap or Infer step. Default is sim for Generator and Bootstrap, and asimov for Infer.', type = str, default = None, choices = ["data", "asimov", "sim"])
  parser.add_argument('--likelihood-type', help = 'Type of likelihood to use for fitting.', type = str, default = "unbinned_extended", choices = ["unbinned_extended", "unbinned", "binned_extended", "binned"])
  parser.add_argument('--submit', help = 'Batch to submit to', type = str, default = None)
  parser.add_argument('--architecture', help = 'Config for running', type = str, default = "configs/architecture/default.yaml")
  parser.add_argument('--points-per-job', help= 'The number of points ran per job', type=int, default=1)
  parser.add_argument('--use-wandb', help='Use wandb for logging.', action='store_true')
  parser.add_argument('--wandb-project-name', help= 'Name of project on wandb', type=str, default="innfer")
  parser.add_argument('--disable-tqdm', help='Disable tqdm when training.', action='store_true')
  parser.add_argument('--dry-run', help='Setup batch submission without running.', action='store_true')
  parser.add_argument('--make-snakemake-inputs', help='Make the snakemake input file', action='store_true')
  parser.add_argument('--hyperparameter-scan-metric', help = 'Colon separated metric name and whether you want max or min, separated by a comma.', type = str, default = "chi_squared_test:total,min")
  parser.add_argument('--plot-2d-unrolled', help='Make 2D unrolled plots when running generator.', action='store_true')
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
      for architecture_ind, architecture in enumerate(GetScanArchitectures(args.architecture, data_output=f"data/{cfg['name']}/HyperparameterScan/{file_name}")):
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
            "plots_output" : f"plots/{cfg['name']}/{file_name}/Generator",
            "scale_to_yield" : "extended" in args.likelihood_type,
            "do_2d_unrolled" : args.plot_2d_unrolled,
          },
          loop = {"file_name" : file_name, "val_ind" : val_ind}
        )

  # Making plots using the network as a generator for individual Y values
  if args.step == "GeneratorSummary":
    print("<< Making plots using the network as a generator summarising all Y values >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}",cfg)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      if len(val_loop) == 1: continue
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
          "plots_output" : f"plots/{cfg['name']}/{file_name}/GeneratorSummary",
          "scale_to_yield" : "extended" in args.likelihood_type,
          "do_2d_unrolled" : args.plot_2d_unrolled,
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

    main(args, default_args)

  else:

    snakemake_file = SetupSnakeMakeFile(args, default_args, main)
    os.system(f"snakemake --cores all --profile htcondor -s '{snakemake_file}' --unlock &> /dev/null")
    os.system(f"snakemake --cores all --profile htcondor -s '{snakemake_file}'")


  print("<< Finished running without error >>")
  end_time = time.time()
  hours, remainder = divmod(end_time-start_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f"<< Time elapsed: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds >>")