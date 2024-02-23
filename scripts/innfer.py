import time
import argparse
import yaml
import os
import sys
import copy
import wandb
import numpy as np
import pyfiglet as pyg
from other_functions import GetValidateLoop, GetPOILoop, GetNuisanceLoop, GetCombinedValidateLoop, GetScanArchitectures, GetYName, MakeDirectories
from pprint import pprint
import pyfiglet as pyg

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c','--cfg', help= 'Config for running',  default=None)
  parser.add_argument('--benchmark', help= 'Run from benchmark scenario',  default=None, choices=["Gaussian","GaussianWithExpBkg","GaussianWithExpBkgVaryingYield","2D","5D","5D+D","12D"])
  parser.add_argument('--architecture', help= 'Config for running',  default="configs/architecture/default.yaml")
  parser.add_argument('--submit', help= 'Batch to submit to', type=str, default=None)
  parser.add_argument('--resubmit-scans', help= 'Resubmit any scan jobs that failed.',  action='store_true')
  parser.add_argument('--step', help= 'Step to run', type=str, default=None, choices=["MakeBenchmark","PreProcess","Train","ValidateGeneration","ValidateInference","Infer"])
  parser.add_argument('--specific-file', help= 'Run for a specific file_name', type=str, default=None)
  parser.add_argument('--specific-val-ind', help= 'Run for a specific indices when doing validation', type=str, default=None)
  parser.add_argument('--specific-scan-ind', help= 'Run for a specific indices when doing scans', type=str, default=None)
  parser.add_argument('--hyperscan-ind', help= 'Add relevant indices on the end of the hyperparameters scan', type=str, default=None)
  parser.add_argument('--scan-points-per-job', help= 'Number of scan points in a single job', type=int, default=100)
  parser.add_argument('--disable-tqdm', help= 'Disable tqdm print out when training.',  action='store_true')
  parser.add_argument('--sge-queue', help= 'Queue for SGE submission', type=str, default="hep.q")
  parser.add_argument('--sub-step', help= 'Sub-step to run for ValidateInference or Infer steps', type=str, default="InitialFit", choices=["InitialFit","Scan","Collect","Plot","Summary","All","Debug"])
  parser.add_argument('--lower-validation-stats', help= 'Lowers the validation stats, so code will run faster.', type=int, default=None)
  parser.add_argument('--do-binned-fit', help= 'Do an extended binned fit instead of an extended unbinned fit.',  action='store_true')
  parser.add_argument('--use-wandb', help='Use wandb for logging.', action='store_true')
  parser.add_argument('--wandb-project-name', help= 'Name of project on wandb', type=str, default="innfer")
  parser.add_argument('--scan-hyperparameters', help='Perform a hyperparameter scan.', action='store_true')
  parser.add_argument('--continue-training', help='Continue training pre-saved NN.', action='store_true')
  parser.add_argument('--skip-generation-correlation', help='Skip all of the 2d correlation ValidateGeneration plots.', action='store_true')
  args = parser.parse_args()

  if args.cfg is None and args.benchmark is None:
    raise ValueError("The --cfg or --benchmark is required.")
  if args.step is None:
    raise ValueError("The --step is required.")

  if args.cfg is not None:
    if "Benchmark" in args.cfg:
      args.benchmark = args.cfg.split("Benchmark_")[1].split(".yaml")[0]

  return args


def main(args, architecture=None):

  if args.benchmark:
    from benchmarks import Benchmarks
    benchmark = Benchmarks(name=args.benchmark)
    if args.step == "MakeBenchmark":
      print("- Making benchmark inputs")

      if args.submit is not None:
        # Submit to batch
        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i])}"
        options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/innfer_{args.step}_{args.benchmark}.sh", "sge_queue":args.sge_queue}
        from batch import Batch
        sub = Batch(options=options)
        sub.Run()
        exit()
      else:
        # Make dataset and config
        benchmark.MakeDataset()
        benchmark.MakeConfig()

    args.cfg = f"configs/run/Benchmark_{args.benchmark}.yaml"

  with open(args.cfg, 'r') as yaml_file:
    cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

  if architecture is None:
    with open(args.architecture, 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
  
  if cfg["preprocess"]["standardise"] == "all":
    cfg["preprocess"]["standardise"] = cfg["variables"] + cfg["pois"]+cfg["nuisances"]

  networks = {}
  parameters = {}
  pp = {}

  for file_name, parquet_name in cfg["files"].items():

    # Skip if condition not met
    if args.specific_file != None and args.specific_file != file_name and args.specific_file != "combined": continue

    # Submit to batch
    if args.step in ["PreProcess","Train"] and args.submit is not None and not args.scan_hyperparameters:
      cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i])} --specific-file={file_name} --disable-tqdm"
      options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/innfer_{args.step}_{cfg['name']}_{file_name}.sh", "sge_queue":args.sge_queue}
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
      pp[file_name].output_dir = f"data/{cfg['name']}/{file_name}/{args.step}"
      pp[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}/{args.step}"
      pp[file_name].Run()

      # Run plots varying the pois across the variables
      for info in GetPOILoop(cfg, pp[file_name].parameters):
        pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="train", extra_name=f'{info["extra_name"]}_train')
        pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="test", extra_name=f'{info["extra_name"]}_test')
        pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="val", extra_name=f'{info["extra_name"]}_val')
        pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="train", extra_name=f'{info["extra_name"]}_train_transformed', transform=True)
        pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="test", extra_name=f'{info["extra_name"]}_test_transformed', transform=True)
        pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="val", extra_name=f'{info["extra_name"]}_val_transformed', transform=True)


      # Run plots varying the nuisances across the variables for each unique value of the pois
      for info in GetNuisanceLoop(cfg, pp[file_name].parameters):
        pp[file_name].PlotX(info["nuisance"], freeze=info["freeze"], dataset="train", extra_name=f'{info["extra_name"]}_train')
        pp[file_name].PlotX(info["nuisance"], freeze=info["freeze"], dataset="test", extra_name=f'{info["extra_name"]}_test')
        pp[file_name].PlotX(info["nuisance"], freeze=info["freeze"], dataset="val", extra_name=f'{info["extra_name"]}_val')
        pp[file_name].PlotX(info["nuisance"], freeze=info["freeze"], dataset="train", extra_name=f'{info["extra_name"]}_train_transformed', transform=True)
        pp[file_name].PlotX(info["nuisance"], freeze=info["freeze"], dataset="test", extra_name=f'{info["extra_name"]}_test_transformed', transform=True)
        pp[file_name].PlotX(info["nuisance"], freeze=info["freeze"], dataset="val", extra_name=f'{info["extra_name"]}_val_transformed', transform=True)          

      # Run plots of the distribution of the context features
      pp[file_name].PlotY(dataset="train")

    ### Training/Loading  of the Networks ###
    if args.step in ["Train","ValidateGeneration","ValidateInference","Infer"]:

      # Load in data parameters
      with open(f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml", 'r') as yaml_file:
        parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

      if not args.do_binned_fit:

        # Load in training architecture if loading in models
        if args.step != "Train":
          with open(f"models/{cfg['name']}/{file_name}_architecture.yaml", 'r') as yaml_file:
            architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
          
        # Set up loop of hyperparameters to try
        if args.scan_hyperparameters:
          run_architectures = GetScanArchitectures(architecture)
        else:
          run_architectures = [architecture]

        # Loop through hyperparameters choices
        for hyper_scan_ind, run_architecture in enumerate(run_architectures):

          # Replace ind for if running on the batch
          if args.hyperscan_ind is not None:
            hyper_scan_ind = args.hyperscan_ind

          # Initiate wandb
          if args.use_wandb and args.step == "Train" and args.submit is None:
            wandb.init(project=args.wandb_project_name, name=f"{cfg['name']}_{file_name}", config=architecture)
            wandb.run.name += f"_{hyper_scan_ind}"

          # Batch submission for hyperparameter scans
          if args.step in ["Train"] and args.submit is not None and args.scan_hyperparameters:
            dump_arch_name = f"configs/scan_architecture/{cfg['name']}/{file_name}/architecture_{hyper_scan_ind}.yaml"
            MakeDirectories(dump_arch_name)
            with open(dump_arch_name, 'w') as file:
              yaml.dump(run_architecture, file)            
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--architecture' not in i])} --specific-file={file_name} --disable-tqdm --architecture={dump_arch_name} --hyperscan-ind={hyper_scan_ind}"
            options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/innfer_{args.step}_{cfg['name']}_{file_name}_hyperscan_{hyper_scan_ind}.sh", "sge_queue":args.sge_queue}
            from batch import Batch
            sub = Batch(options=options)
            sub.Run()
            continue

          if args.scan_hyperparameters:
            print(" - Running with architecture:")
            pprint(run_architecture)

          # Build network
          print("- Building network")
          from network import Network
          networks[file_name] = Network(
            f"{parameters[file_name]['file_location']}/X_train.parquet",
            f"{parameters[file_name]['file_location']}/Y_train.parquet", 
            f"{parameters[file_name]['file_location']}/wt_train.parquet", 
            f"{parameters[file_name]['file_location']}/X_test.parquet",
            f"{parameters[file_name]['file_location']}/Y_test.parquet", 
            f"{parameters[file_name]['file_location']}/wt_test.parquet",
            options=run_architecture)
          
          # Set plotting directory
          networks[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}/{args.step}"
          networks[file_name].data_parameters = parameters[file_name]

          if args.step == "Train":
            # Train model
            print(f"- Training model {file_name}")

            if args.continue_training:
              networks[file_name].Load(name=f"models/{cfg['name']}/{file_name}.h5")
            else:
              networks[file_name].BuildModel()
            networks[file_name].disable_tqdm =  args.disable_tqdm
            networks[file_name].use_wandb = args.use_wandb
            networks[file_name].BuildTrainer()
            networks[file_name].Train()

            # Calculate validation matrics
            AUC = networks[file_name].auc(dataset="test")
            print(">> AUC:", AUC)
            R2 = networks[file_name].r2(dataset="test")
            print(">> R2:")
            for k, v in R2.items(): print(f"  >> {k}: {v}")
            NRMSE = networks[file_name].nrmse(dataset="test")
            print(">> NRMSE:")
            for k, v in NRMSE.items(): print(f"  >> {k}: {v}")

            val_matrics = {
              "AUC": AUC,
              "R2": R2,
              "NRMSE": NRMSE
            }
            if args.use_wandb:
              wandb.log(val_matrics)
              wandb.finish()

            # Check if model should be saved
            sep_score_dump_file = f"models/{cfg['name']}/{file_name}_sep_score.yaml"
            if not args.scan_hyperparameters or not os.path.isfile(sep_score_dump_file):
              save_model = True
            else:
              with open(sep_score_dump_file, 'r') as yaml_file:
                sep_score_dump = yaml.load(yaml_file, Loader=yaml.FullLoader)
              if AUC < sep_score_dump.get("AUC", 0):
                save_model = True
              else:
                save_model = False      

            # Save model
            if save_model:
              networks[file_name].Save(name=f"models/{cfg['name']}/{file_name}.h5")
              with open(f"models/{cfg['name']}/{file_name}_architecture.yaml", 'w') as file:
                yaml.dump(run_architecture, file)
              with open(sep_score_dump_file, 'w') as file:
                yaml.dump(val_matrics, file)

          else:
            # Load model
            print(f"- Loading model {file_name}")
            networks[file_name].Load(name=f"models/{cfg['name']}/{file_name}.h5")

  ### Do validation of trained model ###
  if args.step in ["ValidateGeneration","ValidateInference","Infer"]:

    print("- Performing validation")

    # Set up loop
    if args.step != "Infer":
      file_loop = list(cfg["files"].keys())
      if len(file_loop) > 1:
        file_loop.append("combined")
    else:
      file_loop = ["combined"]

    # Loop through files to do inference on
    for file_name in file_loop:

      # Skip if it doesn't pass the criteria
      if args.specific_file != None and args.specific_file != file_name: continue

      # Define true pdf
      if args.benchmark is not None:
        if file_name == "combined":
          pdf = {name : benchmark.GetPDF(name) for name in cfg["files"].keys()}
        else:
          pdf = {file_name : benchmark.GetPDF(file_name)}
      else:
        pdf = None

      if args.submit is None:

        # Set up validation class
        from validation import Validation
        val = Validation(
          networks if file_name == "combined" else {file_name:networks[file_name]} if not args.do_binned_fit else None, 
          options={
            "infer" : None if args.step != "Infer" else cfg["data_file"],
            "data_key":"val" if not args.do_binned_fit else "full",
            "data_parameters":parameters if file_name == "combined" else {file_name:parameters[file_name]},
            "pois":cfg["pois"],
            "nuisances":cfg["nuisances"],
            "out_dir":f"data/{cfg['name']}/{file_name}/{args.step}",
            "plot_dir":f"plots/{cfg['name']}/{file_name}/{args.step}",
            "model_name":file_name,
            "lower_validation_stats":args.lower_validation_stats if args.step == "ValidateInference" else None,
            "do_binned_fit":args.do_binned_fit,
            "var_and_bins": None if not args.do_binned_fit or "var_and_bins" not in cfg["inference"] else cfg["inference"]["var_and_bins"],
            "calculate_columns_for_plotting": cfg["validation"]["calculate_columns_for_plotting"] if "calculate_columns_for_plotting" in cfg["validation"] else {},
            "validation_options": cfg["inference"] if file_name == "combined" else {}
            }
          )

      # Loop through different combinations of POIs and nuisances
      if file_name == "combined":
        val_loop = GetCombinedValidateLoop(cfg, parameters)
        if args.step == "Infer":
          val_loop = [val_loop[0]]
          val_loop[0]["row"] = None
          val_loop[0]["columns"] = None
      else:
        val_loop = GetValidateLoop(cfg, parameters[file_name])

      for ind, info in enumerate(val_loop):
        
        if args.specific_val_ind != None and args.specific_val_ind != str(ind): continue

        print(f" - Columns: {info['columns']}")
        print(f" - Values: {info['row']}")

        # Validate the INN by sampling through the latent space and using the inverse function to compare to the original datasets
        if args.step == "ValidateGeneration" and not args.do_binned_fit:

          print("- Running validation generation")

          # Submit to batch
          if args.submit is not None:
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind}"
            options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/innfer_{args.step}_{cfg['name']}_{file_name}_{ind}.sh", "sge_queue":args.sge_queue}
            from batch import Batch
            sub = Batch(options=options)
            sub.Run()
            continue

          # Plot synthetic vs simulated comparison
          val.PlotGeneration(info["row"], columns=info["columns"], extra_dir="GenerationTrue1D")
          val.PlotGeneration(info["row"], columns=info["columns"], extra_dir="GenerationTrue1DTransformed", transform=True)
          if len(val.X_columns) > 1 and not args.skip_generation_correlation:
            val.Plot2DPulls(info["row"], columns=info["columns"], extra_dir="GenerationTrue2DPulls")
            val.Plot2DUnrolledGeneration(info["row"], columns=info["columns"], extra_dir="GenerationTrue2D")
            val.Plot2DUnrolledGeneration(info["row"], columns=info["columns"], extra_dir="GenerationTrue2DTransformed", transform=True)
            val.PlotCorrelationMatrix(info["row"], columns=info["columns"], extra_dir="GenerationCorrelation")        

        # Validating the INNs by using it to perform toy inference
        elif args.step == "ValidateInference" or args.step == "Infer":

          if args.step != "Infer":
            if len(info["row"]) == 0: 
              continue

          print("- Running validation inference")

          # Submit to batch
          if args.submit is not None and not args.sub_step == "Scan":
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind}"
            options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/{args.sub_step}/innfer_{args.step}_{args.sub_step}_{cfg['name']}_{file_name}_{ind}.sh", "sge_queue":args.sge_queue}
            from batch import Batch
            sub = Batch(options=options)
            sub.Run()
            continue

          # Build the likelihood likelihood
          if args.submit is None:
            val.BuildLikelihood()

          if args.sub_step == "Debug":
            val.DoDebug(info["row"], columns=info["columns"])

          if args.sub_step in ["InitialFit","All"]:

            print("- Running initial fit")
            # Get the best fit
            val.GetAndDumpBestFit(info["row"], info["initial_best_fit_guess"], ind=ind,  columns=info["columns"])
            # Get scan ranges
            for col in info["columns"]:
              val.GetAndDumpScanRanges(info["row"], col, ind=ind, columns=info["columns"])

          if args.sub_step in ["Scan","All"]:

            print("- Running scans")
            # Load best fit
            with open(f"data/{cfg['name']}/{file_name}/{args.step}/best_fit_{ind}.yaml", 'r') as yaml_file:
              best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

            if args.submit is None:
              val.lkld.best_fit = np.array(best_fit_info["best_fit"])
              val.lkld.best_fit_nll = best_fit_info["best_fit_nll"]

            # Loop through Y values to get a scan for each Y value with the others profiled
            job_ind = 0
            for col in info["columns"]:

              # Load scan ranges
              with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_values_{col}_{ind}.yaml", 'r') as yaml_file:
                scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader) 

              # Run scans
              for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):

                if not (args.specific_scan_ind != None and args.specific_scan_ind != str(job_ind)):
                  if args.submit is None:
                    val.GetAndDumpScan(info["row"], col, scan_value, ind1=ind, ind2=point_ind,  columns=info["columns"])

                if ((point_ind+1) % args.scan_points_per_job == 0) or (point_ind == len(scan_values_info["scan_values"])-1):
                  # Submit to batch
                  if args.submit is not None:
                    cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--specific-scan-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind} --specific-scan-ind={job_ind}"
                    options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/{args.sub_step}/innfer_{args.step}_{args.sub_step}_{cfg['name']}_{file_name}_{ind}_{job_ind}.sh", "sge_queue":args.sge_queue}

                    if args.resubmit_scans:
                      log_file = options["job_name"].replace(".sh","_output.log")
                      finished = False
                      if os.path.exists(log_file):
                        with open(log_file, 'r') as file:
                          for line_number, line in enumerate(file, start=1):
                            if line.strip() == "- Finished running without error":
                              finished = True
                      if finished: 
                        job_ind += 1
                        continue  

                    from batch import Batch
                    sub = Batch(options=options)
                    sub.Run()
                  job_ind += 1

          if args.sub_step in ["Collect","All"]:

            print("- Collecting scans")

            for col in info["columns"]:
              # Load scan ranges
              with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_values_{col}_{ind}.yaml", 'r') as yaml_file:
                scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader) 

              # Load scan results
              scan_results = {"nlls":[],"scan_values":[]}
              for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):      
                with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}_{ind}_{point_ind}.yaml", 'r') as yaml_file:
                  scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                if point_ind == 0:
                  scan_results["row"] = scan_results_info["row"]
                  scan_results["columns"] = scan_results_info["columns"]
                  scan_results["varied_column"] = scan_results_info["varied_column"]
                if None in scan_results_info["nlls"]: continue
                scan_results["scan_values"] += scan_results_info["scan_values"]
                scan_results["nlls"] += scan_results_info["nlls"]

              # Recheck minimum
              min_nll = min(scan_results["nlls"])
              scan_results["nlls"] = [i - min_nll for i in scan_results["nlls"]]
              # Get crossings
              scan_results["crossings"] = val.lkld.FindCrossings(scan_results["scan_values"], scan_results["nlls"], crossings=[1, 2])
              # Dump to yaml
              filename = f"data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}_{ind}.yaml"
              print(f">> Created {filename}")
              with open(filename, 'w') as yaml_file:
                yaml.dump(scan_results, yaml_file, default_flow_style=False)
              # Remove per point yaml files
              for point_ind, scan_value in enumerate(scan_values_info["scan_values"]): 
                os.system(f"rm data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}_{ind}_{point_ind}.yaml")

          if args.sub_step in ["Plot","All"]:

            print("- Plotting scans and comparisons")

            with open(f"data/{cfg['name']}/{file_name}/{args.step}/best_fit_{ind}.yaml", 'r') as yaml_file:
              best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
            for col in info["columns"]:
              with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}_{ind}.yaml", 'r') as yaml_file:
                scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

              val.PlotLikelihood(
                scan_results_info["scan_values"], 
                scan_results_info["nlls"], 
                scan_results_info["row"], 
                col, 
                scan_results_info["crossings"], 
                best_fit_info["best_fit"], 
                true_pdf=pdf, 
                columns=info["columns"],
                extra_dir="LikelihoodScans"
              )

            if not args.do_binned_fit:
              val.PlotComparisons(best_fit_info["row"], best_fit_info["best_fit"], columns=info["columns"], extra_dir="Comparisons")
              val.PlotGeneration(info["row"], columns=info["columns"], sample_row=best_fit_info["best_fit"], extra_dir="GenerationBestFit1D")
              if len(val.X_columns) > 1:
                val.Plot2DUnrolledGeneration(info["row"], columns=info["columns"], sample_row=best_fit_info["best_fit"], extra_dir="GenerationBestFit2D")
            else:
              val.PlotBinned(info["row"], columns=info["columns"], sample_row=best_fit_info["best_fit"], extra_dir="BinnedDistributions")

      if args.step == "ValidateInference" and args.sub_step in ["Summary","All"] and args.submit is None:

        print("- Plotting summary of the validation by inference")

        if len(info["row"]) == 0: 
          continue

        results = {}
        for ind, info in enumerate(val_loop):
          info_name = GetYName(info["row"],purpose="file")
          for col in info["columns"]:
            with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}_{ind}.yaml", 'r') as yaml_file:
              scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
            if col not in results.keys():
              results[col] = {info_name:scan_results_info}
            else:
              results[col][info_name] = scan_results_info

        val.PlotValidationSummary(results, true_pdf=pdf, extra_dir="Summary")


if __name__ == "__main__":

  start_time = time.time()
  title = pyg.figlet_format("INNFER")
  print()
  print(title)

  args = parse_args()
  main(args)
    
  print("- Finished running without error")
  end_time = time.time()
  hours, remainder = divmod(end_time-start_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f"- Time elapsed: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
