import argparse
import time
import os
import sys
import yaml
import pyfiglet as pyg

from useful_functions import CheckRunAndSubmit

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', help = 'Config for running', default = None)
  parser.add_argument('--benchmark', help = 'Run from benchmark scenario', default = None)
  parser.add_argument('--step', help = 'Step to run.', type = str, default = None, choices = ["MakeBenchmark", "PreProcess", "Train", "HyperparameterScan", "Generator", "BootstrapFits", "BootstrapCollect", "BootstrapPlot", "BootstrapSummary", "InferInitialFit", "InferScan", "InferCollect", "InferPlot", "InferSummary"])
  parser.add_argument('--specific', help = 'Specific part of a step to run.', type = str, default = "")
  parser.add_argument('--data-type', help = 'The data type to use when running the Generator, Bootstrap or Infer step. Default is sim for Generator and Bootstrap, and asimov for Infer.', type = str, default = None, choices = ["data", "asimov", "sim"])
  parser.add_argument('--likelihood-type', help = 'Type of likelihood to use for fitting.', type = str, default = "unbinned_extended", choices = ["unbinned_extended", "unbinned", "binned_extended", "binned"])
  parser.add_argument('--submit', help = 'Batch to submit to', type = str, default = None)
  parser.add_argument('--architecture', help = 'Config for running',type = str,   default = "configs/architecture/default.yaml")
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

  # Make the benchmark scenario
  if args.step == "MakeBenchmark":
    print("<< Making benchmark inputs >>")
    from make_benchmark import MakeBenchmark
    run = CheckRunAndSubmit(sys.argv, submit=args.submit, loop = {}, specific = args.specific, job_name = f"jobs/Benchmark_{args.benchmark}/innfer_{args.step}" if ".yaml" not in args.benchmark else f"jobs/Benchmark_{args.benchmark.split('/')[-1].split('.yaml')[0]}/innfer_{args.step}")
    if run:
      mb = MakeBenchmark()
      mb.Configure({"name" : args.benchmark})
      mb.Run()
  else:
    # Load in configuration file
    with open(args.cfg, 'r') as yaml_file:
      cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

  # PreProcess the dataset
  if args.step == "PreProcess":
    print("<< Preprocessing datasets >>")
    from preprocess import PreProcess
    for file_name, parquet_name in cfg["files"].items():
      print(f"* Running file_name={file_name}")
      run = CheckRunAndSubmit(sys.argv, submit=args.submit, loop = {"files" : file_name}, specific = args.specific, job_name = f"jobs/{cfg['name']}/innfer_{args.step}")
      if run:
        pp = PreProcess()
        pp.Configure(
          {
            "cfg" : args.cfg,
            "parquet_file_name" : parquet_name,
            "data_output" : f"data/{cfg['name']}/PreProcess/{file_name}",
            "plots_output" : f"plots/{cfg['name']}/PreProcess/{file_name}",
          }
        )
        pp.Run()

  # Train network
  if args.step == "Train":
    print("<< Training the networks >>")
  #  from train import Train
    for file_name, parquet_name in cfg["files"].items():
      run = CheckRunAndSubmit(sys.argv, loop = {"files" : file_name}, specific = args.specific, job_name = f"jobs/{cfg['name']}/innfer_{args.step}")
  #    if run:
  #      t = Train()
  #      t.Configure({"cfg" : cfg})
  #      t.Run()


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