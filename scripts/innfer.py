import argparse
import yaml
import os
import sys
import numpy as np
from other_functions import GetValidateLoop, GetPOILoop, GetNuisanceLoop

print("Running INNFER")

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cfg', help= 'Config for running',  default=None)
parser.add_argument('--benchmark', help= 'Run from benchmark scenario',  default=None, choices=["Gaussian","GaussianWithExpBkg"])
parser.add_argument('--architecture', help= 'Config for running',  default="configs/architecture/default.yaml")
parser.add_argument('--submit', help= 'Batch to submit to', type=str, default=None)
parser.add_argument('--step', help= 'Step to run', type=str, default=None, choices=["PreProcess","Train","ValidateGeneration","ValidateInference","Infer"])
parser.add_argument('--specific-file', help= 'Run for a specific file_name', type=str, default=None)
parser.add_argument('--specific-val-ind', help= 'Run for a specific indices when doing validation', type=str, default=None)
parser.add_argument('--specific-scan-ind', help= 'Run for a specific indices when doing scans', type=str, default=None)
parser.add_argument('--scan-points-per-job', help= 'Number of scan points in a single job', type=int, default=10)
parser.add_argument('--disable-tqdm', help= 'Disable tqdm print out when training.',  action='store_true')
parser.add_argument('--sge-queue', help= 'Queue for SGE submission', type=str, default="hep.q")
parser.add_argument('--sub-step', help= 'Sub-step to run for ValidateInference or Infer steps', type=str, default="InitialFit", choices=["InitialFit","Scan","Collect","Plot"])
args = parser.parse_args()

if args.cfg is None and args.benchmark is None:
  raise ValueError("The --cfg or --benchmark is required.")
if args.step is None:
  raise ValueError("The --step is required.")

if args.benchmark:
  from benchmarks import Benchmarks
  benchmark = Benchmarks(name=args.benchmark)
  if args.step == "PreProcess":
    benchmark.MakeDataset()
    benchmark.MakeConfig()
  args.cfg = f"configs/run/Benchmark_{args.benchmark}.yaml"

with open(args.cfg, 'r') as yaml_file:
  cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

with open(args.architecture, 'r') as yaml_file:
  architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

if not os.path.isdir(f"data/{cfg['name']}"): os.system(f"mkdir data/{cfg['name']}")
if not os.path.isdir(f"models/{cfg['name']}"): os.system(f"mkdir models/{cfg['name']}")
if not os.path.isdir(f"plots/{cfg['name']}"): os.system(f"mkdir plots/{cfg['name']}")
for file_name, _ in cfg["files"].items():
  if not os.path.isdir(f"data/{cfg['name']}/{file_name}"): os.system(f"mkdir data/{cfg['name']}/{file_name}")
  if not os.path.isdir(f"plots/{cfg['name']}/{file_name}"): os.system(f"mkdir plots/{cfg['name']}/{file_name}")  

if cfg["preprocess"]["standardise"] == "all":
  cfg["preprocess"]["standardise"] = cfg["variables"] + cfg["pois"]+cfg["nuisances"]

networks = {}
parameters = {}
pp = {}

for file_name, parquet_name in cfg["files"].items():

  # Skip if condition not met
  if args.specific_file != None and args.specific_file != file_name: continue

  # Submit to batch
  if args.step in ["PreProcess","Train"] and args.submit is not None:
    cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i])} --specific-file={file_name} --disable-tqdm"
    options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/innfer_{args.step.lower()}_{cfg['name']}_{file_name}.sh", "sge_queue":args.sge_queue}
    from batch import Batch
    sub = Batch(options=options)
    sub.Run()
    continue

  ### PreProcess Data ###
  if args.step == "PreProcess":
    print("- Preprocessing data")

    # PreProcess the dataset
    from preprocess import PreProcess
    pp[file_name] = PreProcess(parquet_name, cfg["variables"], cfg["pois"]+cfg["nuisances"], options=cfg["preprocess"])
    pp[file_name].output_dir = f"data/{cfg['name']}/{file_name}"
    pp[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}"
    pp[file_name].Run()

    # Run plots varying the pois across the variables
    for info in GetPOILoop(cfg, pp[file_name].parameters):
      pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="train", extra_name=info["extra_name"])
        
    # Run plots varying the nuisances across the variables for each unique value of the pois
    for info in GetNuisanceLoop(cfg, pp[file_name].parameters):
      pp[file_name].PlotX(info["nuisance"], freeze=info["freeze"], dataset="train", extra_name=info["extra_name"])
        
    # Run plots of the distribution of the context features
    pp[file_name].PlotY(dataset="train")


  ### Training, validation and inference ####
  if args.step in ["Train","ValidateGeneration","ValidateInference","Infer"]:

    with open(f"data/{cfg['name']}/{file_name}/parameters.yaml", 'r') as yaml_file:
      parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if args.submit is None:

      if args.step != "Train":
        with open(f"models/{cfg['name']}/{file_name}_architecture.yaml", 'r') as yaml_file:
          architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

      print("- Building network")
      from network import Network
      networks[file_name] = Network(
        f"data/{cfg['name']}/{file_name}/X_train.parquet",
        f"data/{cfg['name']}/{file_name}/Y_train.parquet", 
        f"data/{cfg['name']}/{file_name}/wt_train.parquet", 
        f"data/{cfg['name']}/{file_name}/X_test.parquet",
        f"data/{cfg['name']}/{file_name}/Y_test.parquet", 
        f"data/{cfg['name']}/{file_name}/wt_test.parquet",
        options=architecture)
      
      networks[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}"

      ### Train or load networks ###
      if args.step == "Train":
        print("- Training model")
        networks[file_name].BuildModel()
        networks[file_name].disable_tqdm =  args.disable_tqdm
        networks[file_name].BuildTrainer()
        networks[file_name].Train()
        networks[file_name].Save(name=f"models/{cfg['name']}/{file_name}.h5")
        with open(f"models/{cfg['name']}/{file_name}_architecture.yaml", 'w') as file:
          yaml.dump(architecture, file)
      else:
        print("- Loading model")
        networks[file_name].Load(name=f"models/{cfg['name']}/{file_name}.h5")
        networks[file_name].data_parameters = parameters[file_name]    
  
    ### Do validation of trained model ###
    if args.step in ["ValidateGeneration","ValidateInference"]:

      print("- Performing validation")

      if args.submit is None:
        from validation import Validation
        val = Validation(
          networks[file_name], 
          options={
            "data_parameters":parameters[file_name],
            "data_dir":f"data/{cfg['name']}/{file_name}",
            "plot_dir":f"plots/{cfg['name']}/{file_name}",
            "model_name":file_name,
            }
          )

      for ind, info in enumerate(GetValidateLoop(cfg, parameters[file_name])):

        if args.specific_val_ind != None and args.specific_val_ind != str(ind): continue

        print(f" - Columns: {info['columns']}")
        print(f" - Values: {info['row']}")

        if args.step == "ValidateGeneration":

          print("- Running generation plot")

          # Submit to batch
          if args.submit is not None:
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind'not in i])} --specific-file={file_name} --specific-val-ind={ind}"
            options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/innfer_{args.step.lower()}_{cfg['name']}_{file_name}_{ind}.sh", "sge_queue":args.sge_queue}
            from batch import Batch
            sub = Batch(options=options)
            sub.Run()
            continue

          # Plot synthetic vs simulated comparison
          val.PlotGeneration(info["row"], columns=info["columns"])

        elif args.step == "ValidateInference":

          print("- Running validation inference")

          # Submit to batch
          if args.submit is not None and not args.sub_step == "Scan":
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind'not in i])} --specific-file={file_name} --specific-val-ind={ind}"
            options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/innfer_{args.step.lower()}_{args.sub_step.lower()}_{cfg['name']}_{file_name}_{ind}.sh", "sge_queue":args.sge_queue}
            from batch import Batch
            sub = Batch(options=options)
            sub.Run()
            continue

          # Build an unbinned likelihood for a single model with no constraints
          if args.submit is None:

            val.BuildLikelihood()

            if args.sub_step == "InitialFit":

              print("- Running initial fit")
              # Get the best fit
              val.GetAndDumpBestFit(info["row"], info["initial_best_fit_guess"], ind=ind)
              # Get scan ranges
              for col in info["columns"]:
                val.GetAndDumpScanRanges(info["row"], col, ind=ind)


          if args.sub_step == "Scan":

            print("- Running scans")
            # Load best fit
            with open(f"data/{cfg['name']}/{file_name}/best_fit_{ind}.yaml", 'r') as yaml_file:
              best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

            if args.submit is None:
              val.lkld.best_fit = np.array(best_fit_info["best_fit"])
              val.lkld.best_fit_nll = best_fit_info["best_fit_nll"]

            # Loop through Y values to get a scan for each Y value with the others profiled
            for col in info["columns"]:

              # Load scan ranges
              with open(f"data/{cfg['name']}/{file_name}/scan_values_{col}_{ind}.yaml", 'r') as yaml_file:
                scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader) 

              # Run scans
              job_ind = 0
              for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):

                if not (args.specific_scan_ind != None and args.specific_scan_ind != str(job_ind)):
                  if args.submit is None:
                    val.GetAndDumpNLL(info["row"], col, scan_value, ind1=ind, ind2=point_ind)

                if ((point_ind+1) % args.scan_points_per_job == 0) or (point_ind == len(scan_values_info["scan_values"])-1):
                  # Submit to batch
                  if args.submit is not None:
                    cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind'not in i and '--specific-scan-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind} --specific-scan-ind={job_ind}"
                    options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/innfer_{args.step.lower()}_{args.sub_step.lower()}_{cfg['name']}_{file_name}_{ind}_{job_ind}.sh", "sge_queue":args.sge_queue}
                    from batch import Batch
                    sub = Batch(options=options)
                    sub.Run()
                  job_ind += 1

          if args.sub_step == "Collect":

            print("- Collecting scans")

            for col in info["columns"]:
              # Load scan ranges
              with open(f"data/{cfg['name']}/{file_name}/scan_values_{col}_{ind}.yaml", 'r') as yaml_file:
                scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader) 

              # Load scan results
              scan_results = {"nlls":[],"scan_values":[]}
              for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):      
                with open(f"data/{cfg['name']}/{file_name}/scan_results_{col}_{ind}_{point_ind}.yaml", 'r') as yaml_file:
                  scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                if point_ind == 0:
                  scan_results["row"] = scan_results_info["row"]
                  scan_results["columns"] = scan_results_info["columns"]
                  scan_results["varied_column"] = scan_results_info["varied_column"]
                scan_results["scan_values"] += scan_results_info["scan_values"]
                scan_results["nlls"] += scan_results_info["nlls"]

              # Recheck minimum
              min_nll = min(scan_results["nlls"])
              scan_results["nlls"] = [i - min_nll for i in scan_results["nlls"]]
              # Get crossings
              scan_results["crossings"] = val.lkld.FindCrossings(scan_results["scan_values"], scan_results["nlls"], crossings=[1, 2])
              # Dump to yaml
              filename = f"data/{cfg['name']}/{file_name}/scan_results_{col}_{ind}.yaml"
              print(f">> Created {filename}")
              with open(filename, 'w') as yaml_file:
                yaml.dump(scan_results, yaml_file, default_flow_style=False)
              # Remove per point yaml files
              for point_ind, scan_value in enumerate(scan_values_info["scan_values"]): 
                os.system(f"rm data/{cfg['name']}/{file_name}/scan_results_{col}_{ind}_{point_ind}.yaml")

          if args.sub_step == "Plot":

            print("- Plotting scans and comparisons")

            with open(f"data/{cfg['name']}/{file_name}/best_fit_{ind}.yaml", 'r') as yaml_file:
              best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
            for col in info["columns"]:
              with open(f"data/{cfg['name']}/{file_name}/scan_results_{col}_{ind}.yaml", 'r') as yaml_file:
                scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

              val.PlotLikelihood(scan_results_info["scan_values"], scan_results_info["nlls"], scan_results_info["row"], col, scan_results_info["crossings"], best_fit_info["best_fit"], true_pdf=(None if args.benchmark is None else benchmark.GetPDF))
            val.PlotComparisons(best_fit_info["row"], best_fit_info["best_fit"])


# Do we want to validate the combined model? If so, may be better to validate from the lkld? Although then sampling is less meaningful, although we could store the models and do this
"""
### Do inference on data ###
if args.step == "Infer" and cfg["data_file"] is not None:
  print("- Performing inference")

  # Likelihood
  lkld = Likelihood(
    {"pdfs":{k:networks[k] for k in cfg["files"]}}, 
    type = "unbinned", 
    data_parameters = {k: parameters[k] for k in cfg["files"]},
    parameters = cfg["inference"],
  )

  # Data

  # Need an initial guess (central values of Y and 0 for rate?)

  print(">> Getting best fit")
  #lkld.GetBestFit(X, np.array(initial_guess), wts=wt)

  print(">> Making scan")
  # Do we do this for all columns or just pois? Probably all columns, then we can make a summary plot.
"""