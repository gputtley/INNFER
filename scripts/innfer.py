import argparse
import time
import os
import sys
import yaml
import copy
import pyfiglet as pyg

#from useful_functions import CheckRunAndSubmit, GetScanArchitectures, GetValidateInfo
from useful_functions import GetScanArchitectures, GetValidateInfo
from useful_classes import CheckRunAndSubmit

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', help = 'Config for running', default = None)
  parser.add_argument('--benchmark', help = 'Run from benchmark scenario', default = None)
  parser.add_argument('--step', help = 'Step to run.', type = str, default = None, choices = ["MakeBenchmark", "PreProcess", "Train",  "PerformanceMetrics", "HyperparameterScan", "HyperparameterScanCollect", "Generator", "GeneratorSummary", "BootstrapFits", "BootstrapCollect", "BootstrapPlot", "BootstrapSummary", "InferInitialFit", "InferScan", "InferCollect", "InferPlot", "InferSummary"])
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
  parser.add_argument('--hyperparameter-scan-metric', help = 'Colon separated metric name and whether you want max or min, separated by a comma.', type = str, default = "chi_squared_test:total,min")
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

  return args

def main(args):

  # Initiate class that determines whether the relevant loop should be ran, skipped or submitted
  submit = CheckRunAndSubmit(
    sys.argv, 
    submit = args.submit,
    specific = args.specific,
    points_per_job = args.points_per_job,
    dry_run = args.dry_run,
  )

  # Make the benchmark scenario
  if args.step == "MakeBenchmark":
    print("<< Making benchmark inputs >>")
    from make_benchmark import MakeBenchmark
    submit.job_name = f"jobs/Benchmark_{args.benchmark}/innfer_{args.step}" if ".yaml" not in args.benchmark else f"jobs/Benchmark_{args.benchmark.split('/')[-1].split('.yaml')[0]}/innfer_{args.step}"
    run = submit.Run(loop = {})
    if run:
      mb = MakeBenchmark()
      mb.Configure({"name" : args.benchmark})
      mb.Run()
  else:
    # Load in configuration file
    with open(args.cfg, 'r') as yaml_file:
      cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    submit.job_name = f"jobs/{cfg['name']}/innfer_{args.step}"

  # PreProcess the dataset
  if args.step == "PreProcess":
    print("<< Preprocessing datasets >>")
    for file_name, parquet_name in cfg["files"].items():
      run = submit.Run(loop = {"file_name" : file_name})
      if run:
        print(f"* Running file_name={file_name}")
        from preprocess import PreProcess
        pp = PreProcess()
        pp.Configure(
          {
            "cfg" : args.cfg,
            "file_name" : file_name,
            "parquet_file_name" : parquet_name,
            "data_output" : f"data/{cfg['name']}/{file_name}/PreProcess",
            "plots_output" : f"plots/{cfg['name']}/{file_name}/PreProcess",
          }
        )
        pp.Run()
    submit.Sweep()

  # Train network
  if args.step == "Train":
    print("<< Training the networks >>")
    for file_name, parquet_name in cfg["files"].items():
      run = submit.Run(loop = {"file_name" : file_name})
      if run:
        print(f"* Running file_name={file_name}")
        from train import Train
        t = Train()
        t.Configure(
          {
            "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
            "architecture" : args.architecture,
            "data_output" : f"models/{cfg['name']}/{file_name}",
            "plots_output" : f"plots/{cfg['name']}/{file_name}/Train/",
            "disable_tqdm" : args.disable_tqdm,
          }
        )
        t.Run()
    submit.Sweep()


  # Get performance metrics
  if args.step == "PerformanceMetrics":
    print("<< Getting the performance metrics of the trained networks >>")
    for file_name, parquet_name in cfg["files"].items():
      run = submit.Run(loop = {"file_name" : file_name})
      if run:
        print(f"* Running file_name={file_name}")
        from performance_metrics import PerformanceMetrics
        pf = PerformanceMetrics()
        pf.Configure(
          {
            "model" : f"models/{cfg['name']}/{file_name}/{file_name}.h5",
            "architecture" : f"models/{cfg['name']}/{file_name}/{file_name}_architecture.yaml",
            "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
            "data_output" : f"data/{cfg['name']}/{file_name}/PerformanceMetrics",
          }
        )
        pf.Run()
    submit.Sweep()


  # Perform a hyperparameter scan
  if args.step == "HyperparameterScan":
    print("<< Running a hyperparameter scan >>")
    for file_name, parquet_name in cfg["files"].items():
      for architecture_ind, architecture in enumerate(GetScanArchitectures(args.architecture, data_output=f"data/{cfg['name']}/HyperparameterScan/{file_name}")):
        run = submit.Run(loop = {"file_name" : file_name, "architecture_ind" : architecture_ind})
        if run:
          print(f"* Running file_name={file_name};architecture_ind={architecture_ind}")
          from hyperparameter_scan import HyperparameterScan
          hs = HyperparameterScan()
          hs.Configure(
            {
              "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
              "architecture" : architecture,
              "data_output" : f"data/{cfg['name']}/{file_name}/HyperparameterScan",
              "use_wandb" : args.use_wandb,
              "wandb_project_name" : args.wandb_project_name,
              "wandb_submit_name" : f"{cfg['name']}_{file_name}",
              "disable_tqdm" : args.disable_tqdm,
              "save_extra_name" : f"_{architecture_ind}",
            }
          )
          hs.Run()
    submit.Sweep()

  # Perform a hyperparameter scan
  if args.step == "HyperparameterScanCollect":
    print("<< Collecting hyperparameter scan >>")
    for file_name, parquet_name in cfg["files"].items():
      run = submit.Run(loop = {"file_name" : file_name})
      if run:
        print(f"* Running file_name={file_name}")
        from hyperparameter_scan_collect import HyperparameterScanCollect
        hsc = HyperparameterScanCollect()
        hsc.Configure(
          {
            "parameters" : f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml",
            "save_extra_names" : [f"_{architecture_ind}" for architecture_ind in range(len(GetScanArchitectures(args.architecture, write=False)))],
            "data_input" : f"data/{cfg['name']}/{file_name}/HyperparameterScan",
            "data_output" : f"models/{cfg['name']}/{file_name}",
            "metric" : args.hyperparameter_scan_metric
          }
        )
        hsc.Run()
    submit.Sweep()


  # Making plots using the network as a generator for individual Y values
  if args.step == "Generator":
    print("<< Making plots using the network as a generator for individual Y values >>")
    val_loop_info = GetValidateInfo(f"data/{cfg['name']}", f"models/{cfg['name']}",cfg)
    for file_name, val_loop in val_loop_info["val_loops"].items():
      for val_ind, val_info in enumerate(val_loop):
        run = submit.Run(loop = {"file_name" : file_name, "val_ind" : val_ind})
        if run:
          print(f"* Running file_name={file_name};val_ind={val_ind}")
          from generator import Generator
          g = Generator()
          g.Configure(
            {
              "Y_sim" : val_info["row"],
              "Y_synth" : val_info["row"],
              "parameters" : val_loop_info["parameters"][file_name],
              "model" : val_loop_info["models"][file_name],
              "architecture" : val_loop_info["architectures"][file_name],
              "yield_function" : "default",
              "pois" : cfg["pois"],
              "nuisances" : cfg["nuisances"],
              "plots_output" : f"plots/{cfg['name']}/{file_name}/Generator/",
              "scale_to_yield" : "extended" in args.likelihood_type,
            }
          )
          g.Run()
    submit.Sweep()


if __name__ == "__main__":

  start_time = time.time()
  title = pyg.figlet_format("INNFER")
  print()
  print(title)

  args = parse_args()
  main(args)

  print("<< Finished running without error >>")
  end_time = time.time()
  hours, remainder = divmod(end_time-start_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f"<< Time elapsed: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds >>")