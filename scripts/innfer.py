import time
import argparse
import yaml
import os
import sys
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
  parser.add_argument('--step', help= 'Step to run', type=str, default=None, choices=["MakeBenchmark","PreProcess","Train","ValidateGeneration","ValidateInference","Infer","SnakeMake","MakePDFs"])
  parser.add_argument('--specific-file', help= 'Run for a specific file_name', type=str, default=None)
  parser.add_argument('--specific-val-ind', help= 'Run for a specific indices when doing validation', type=str, default=None)
  parser.add_argument('--specific-scan-ind', help= 'Run for a specific indices when doing scans', type=str, default=None)
  parser.add_argument('--specific-bootstrap-ind', help= 'Run for a specific indices when doing bootstrapping', type=str, default=None)
  parser.add_argument('--hyperscan-ind', help= 'Add relevant indices on the end of the hyperparameters scan', type=str, default=None)
  parser.add_argument('--scan-points-per-job', help= 'Number of scan points in a single job', type=int, default=1)
  parser.add_argument('--disable-tqdm', help= 'Disable tqdm print out when training.',  action='store_true')
  parser.add_argument('--sge-queue', help= 'Queue for SGE submission', type=str, default="hep.q")
  parser.add_argument('--sub-step', help= 'Sub-step to run for ValidateInference or Infer steps', type=str, default="All")
  parser.add_argument('--lower-validation-stats', help= 'Lowers the validation stats, so code will run faster.', type=int, default=None)
  parser.add_argument('--likelihood-type', help= 'Type of likelihood', type=str, default="unbinned_extended", choices=["unbinned_extended", "unbinned", "binned_extended", "binned"])
  parser.add_argument('--minimisation-method', help= 'Method for minimisation', type=str, default="nominal", choices=["nominal", "low_stat_high_stat", "quadratic"])
  parser.add_argument('--use-wandb', help='Use wandb for logging.', action='store_true')
  parser.add_argument('--wandb-project-name', help= 'Name of project on wandb', type=str, default="innfer")
  parser.add_argument('--scan-hyperparameters', help='Perform a hyperparameter scan.', action='store_true')
  parser.add_argument('--continue-training', help='Continue training pre-saved NN.', action='store_true')
  parser.add_argument('--skip-generation-correlation', help='Skip all of the 2d correlation ValidateGeneration plots.', action='store_true')
  parser.add_argument('--number-of-bootstraps', help= 'The number of bootstrapped resamples datasets to do ValidateInference for.', type=int, default=100)
  parser.add_argument('--data-type', help= 'The data type to use when running the Infer step', type=str, default="asimov", choices=["data","asimov","sim"])
  parser.add_argument('--number-of-asimov-events', help= 'The number of asimov events generated', type=int, default=10**6)
  parser.add_argument('--freeze', help= 'Comma and colon separated list of conditional variables to freeze in the fit', type=str, default="")
  parser.add_argument('--number-of-scan-points', help= 'The number of scan points ran', type=int, default=17)
  parser.add_argument('--sigma-between-scan-points', help= 'The number of estimated sigmas between each scan point', type=float, default=0.4)
  parser.add_argument('--scale-to-n-eff', help='Scale to the number of effective events.', action='store_true')
  parser.add_argument('--extra-dir-name', help= 'Extra name on the directory.', type=str, default="")
  args = parser.parse_args()

  if args.cfg is None and args.benchmark is None:
    raise ValueError("The --cfg or --benchmark is required.")
  if args.step is None:
    raise ValueError("The --step is required.")

  if args.cfg is not None:
    if "Benchmark" in args.cfg:
      args.benchmark = args.cfg.split("Benchmark_")[1].split(".yaml")[0]

  if os.path.exists(f"configs/run/{args.cfg}"):
    args.cfg = f"configs/run/{args.cfg}"

  if args.architecture is not None:
    if os.path.exists(f"configs/architecture/{args.architecture}"):
      args.architecture = f"configs/architecture/{args.architecture}"

  ValidateInference_substeps = ["InitialFits","Collect","Plot","Summary","All","Debug"]
  if args.step == "ValidateInference" and args.sub_step not in ValidateInference_substeps:
    raise ValueError(f"For --step={args.step}, --sub-step must be either {', '.join(ValidateInference_substeps[:-1])} or {ValidateInference_substeps[-1]}.")

  Infer_substeps = ["InitialFit","Scan","Collect","Plot","Summary","All","Debug"]
  if args.step == "Infer" and args.sub_step not in Infer_substeps:
    raise ValueError(f"For --step={args.step}, --sub-step must be either {', '.join(Infer_substeps[:-1])} or {Infer_substeps[-1]}.")  

  # Make sure it is odd
  if args.number_of_scan_points % 2 == 0:
    args.number_of_scan_points += 1

  return args


def main(args, architecture=None):

  if args.step in ["SnakeMakePre","SnakeMakePost"]: rules = []

  if args.benchmark:
    from benchmarks import Benchmarks
    benchmark = Benchmarks(name=args.benchmark)
    if args.step in ["MakeBenchmark","SnakeMakePre"]:
      print("- Making benchmark inputs")

      if args.submit is not None or args.step == "SnakeMakePre":
        # Submit to batch
        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i])}"
        options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/innfer_{args.step}_{args.benchmark}.sh", "sge_queue":args.sge_queue}
        from batch import Batch
        sub = Batch(options=options)
        if args.step == "SnakeMakePre":
          cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--step' not in i])} --specific-file={file_name} --step=MakeBenchmark" 
          options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/MakeBenchmark/innfer_MakeBenchmark_{cfg['name']}_{file_name}.sh"}
          from batch import Batch
          sub = Batch(options=options)
          sub._CreateBatchJob([cmd])
          rules += [
            "rule MakeBenchmark:",
            " output:",
            f"   'configs/run/Benchmark_{args.benchmark}.yaml'",
            "  threads: 2",
            "  params:",
            "    runtime=3600",
            " shell:",
            f"   'bash jobs/innfer_MakeBenchmark_{args.benchmark}.sh'",
            "",
          ]
        if args.submit is not None:
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

  freeze = {i.split(":")[0]:float(i.split(":")[1]) for i in args.freeze.split(",")} if args.freeze != "" else {}

  networks = {}
  parameters = {}
  pp = {}

  for file_name, parquet_name in cfg["files"].items():

    # Skip if condition not met
    if args.specific_file != None and args.specific_file != file_name and args.specific_file != "combined": continue

    # Submit to batch
    if args.step in ["PreProcess","Train","SnakeMakePre","SnakeMakePost"] and (args.submit is not None or args.step in ["SnakeMakePre","SnakeMakePost"]) and not args.scan_hyperparameters:
      if not args.step in ["SnakeMakePre","SnakeMakePost"]:
        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i])} --specific-file={file_name} --disable-tqdm"
        options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/innfer_{args.step}_{cfg['name']}_{file_name}.sh", "sge_queue":args.sge_queue}
        from batch import Batch
        sub = Batch(options=options)
        sub.Run()
        continue
      elif args.step == "SnakeMakePre":
        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--step' not in i])} --specific-file={file_name} --step=PreProcess" 
        options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/PreProcess/innfer_PreProcess_{cfg['name']}_{file_name}.sh"}
        from batch import Batch
        sub = Batch(options=options)
        sub._CreateBatchJob([cmd])
        rules += [
          f"rule PreProcess_{file_name}:",
          "  input:",
          f"    '{args.cfg}'",
          "  output:",
          f"    'data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml'",
          "  threads: 4",
          "  params:",
          "    runtime=3600",
          "  shell:",
          f"    'bash jobs/{cfg['name']}/PreProcess/innfer_PreProcess_{cfg['name']}_{file_name}.sh'",
          "",
        ]
      elif args.step == "SnakeMakePost" and not args.likelihood_type in ["binned","binned_extended"]:
        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--step' not in i])} --specific-file={file_name} --disable-tqdm --step=Train" 
        options = {"job_name": f"jobs/{cfg['name']}/Train/innfer_Train_{cfg['name']}_{file_name}.sh"}
        from batch import Batch
        sub = Batch(options=options)
        sub._CreateBatchJob([cmd])
        rules += [
          f"rule Train_{file_name}:",
          "  input:",
          f"    'data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml'",
          "  output:",
          f"    'models/{cfg['name']}/{file_name}.h5'",
          "  threads: 1",
          "  params:",
          "    runtime=10800,",
          "    gpu=1",
          "  shell:",
          f"    'bash jobs/{cfg['name']}/Train/innfer_Train_{cfg['name']}_{file_name}.sh'",
          "",
        ]

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
    if args.step in ["Train","ValidateGeneration","ValidateInference","Infer", "SnakeMakePost"]:

      # Load in data parameters
      with open(f"data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml", 'r') as yaml_file:
        parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

      if not args.step == "SnakeMakePost":

        if not args.likelihood_type in ["binned_extended","binned"]:

          # Load in training architecture if loading in models
          if not args.step in ["Train","SnakeMakePost"]:
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
              AUC = networks[file_name].auc(dataset="train")
              print(">> Train AUC:", AUC)
              AUC = networks[file_name].auc(dataset="test")
              print(">> Test AUC:", AUC)
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

  if args.extra_dir_name != "":
    cfg["name"] += args.extra_dir_name

  # Set up loop
  file_loop = list(cfg["files"].keys())
  if len(file_loop) > 1:
    file_loop.append("combined")

  ### Do validation of trained model ###
  if args.step in ["ValidateGeneration","ValidateInference","Infer","SnakeMakePost"]:

    # Loop through files to do inference on
    for file_name in file_loop:

      # Skip if it doesn't pass the criteria
      if args.specific_file != None and args.specific_file != file_name: continue
      if args.step == "Infer" and args.data_type == "data" and args.file_name != "combined": continue

      """
      # Define true pdf
      if args.benchmark is not None:
        if file_name == "combined":
          pdf = {name : benchmark.GetPDF(name) for name in cfg["files"].keys()}
        else:
          pdf = {file_name : benchmark.GetPDF(file_name)}
      else:
        pdf = None
      """
        
      if args.submit is None and not args.step == "SnakeMakePost":

        # Set up validation class
        from validation_and_inference import ValidationAndInference
        val = ValidationAndInference(
          networks if file_name == "combined" else {file_name:networks[file_name]} if not args.likelihood_type in ["binned_extended","binned"] else None, 
          options={
            "infer" : cfg["data_file"] if "data_file" in cfg else None,
            "data_key":"val" if not args.likelihood_type in ["binned_extended","binned"] else "full",
            "data_parameters":parameters if file_name == "combined" else {file_name:parameters[file_name]},
            "pois":cfg["pois"],
            "nuisances":cfg["nuisances"],
            "out_dir":f"data/{cfg['name']}/{file_name}/{args.step}",
            "plot_dir":f"plots/{cfg['name']}/{file_name}/{args.step}",
            "model_name":file_name,
            "lower_validation_stats":args.lower_validation_stats if args.step == "ValidateInference" else None,
            "likelihood_type":args.likelihood_type,
            "var_and_bins": None if not args.likelihood_type in ["binned_extended","binned"] or "var_and_bins" not in cfg["inference"] else cfg["inference"]["var_and_bins"],
            "calculate_columns_for_plotting": cfg["validation"]["calculate_columns_for_plotting"] if "calculate_columns_for_plotting" in cfg["validation"] else {},
            "validation_options": cfg["inference"] if file_name == "combined" else {},
            "number_of_asimov_events": args.number_of_asimov_events,
            }
          )
        
        # Set data type to use for validation and inference steps and other items
        if args.step == "Infer" and args.likelihood_type not in ["binned_extended","binned"]:
          val.data_type = args.data_type
        else:
          val.data_type = "sim"
          #val.data_type = "asimov"


      # Loop through different combinations of POIs and nuisances
      if file_name == "combined":
        val_loop = GetCombinedValidateLoop(cfg, parameters)
        if args.step == "Infer" and args.data_type == "data":
          val_loop = [val_loop[0]]
          val_loop[0]["row"] = None
          val_loop[0]["columns"] = None
      else:
        val_loop = GetValidateLoop(cfg, parameters[file_name])

      for ind, info in enumerate(val_loop):
        
        if args.specific_val_ind != None and args.specific_val_ind != str(ind): continue

        print(f" - Columns: {info['columns']}")
        print(f" - Values: {info['row']}")

        # Setup name extensions
        if args.data_type =="data":
          val_name_ext = ""
        else:
          val_name_ext = f"_val_{ind}"

        # Validate the INN by sampling through the latent space and using the inverse function to compare to the original datasets
        if args.step in ["ValidateGeneration","SnakeMakePost"] and not args.likelihood_type in ["binned_extended","binned"]:

          print("- Running validation generation")
          dummy_file_name = f"data/{cfg['name']}/{file_name}/ValidateGeneration/{file_name}{val_name_ext}.txt"

          # Submit to batch
          if args.submit is not None or args.step == "SnakeMakePost":
            if not args.step == "SnakeMakePost":
              cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind}"
              options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/innfer_{args.step}_{cfg['name']}_{file_name}{val_name_ext}.sh", "sge_queue":args.sge_queue}
              from batch import Batch
              sub = Batch(options=options)
              sub.Run()
              continue
            else:
              cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--step' not in i])} --specific-file={file_name} --specific-val-ind={ind} --step=ValidateGeneration" 
              options = {"job_name": f"jobs/{cfg['name']}/ValidateGeneration/innfer_ValidateGeneration_{cfg['name']}_{file_name}{val_name_ext}.sh"}
              from batch import Batch
              sub = Batch(options=options)
              sub._CreateBatchJob([cmd])
              model_inputs = [f"    'models/{cfg['name']}/{file_name}.h5'"]
              if file_name == "combined":
                model_inputs = []
                for ind_for_val_gen, file_name_for_val_gen in enumerate(file_loop):
                  if file_name_for_val_gen == "combined": continue
                  name_for_val_gen = f"    'models/{cfg['name']}/{file_name_for_val_gen}.h5'"
                  if not ind_for_val_gen == len(file_loop) - 2:
                    name_for_val_gen += ","
                  model_inputs.append(name_for_val_gen)
              rules += [
                f"rule ValidateGeneration_{file_name}{val_name_ext}:",
                "  input:"
              ]
              rules += model_inputs
              rules += [
                "  output:",
                f"    '{dummy_file_name}'",
                "  threads: 2",
                "  params:",
                "    runtime=10800,",
                "    has_avx=1",
                "  shell:",
                f"    'bash jobs/{cfg['name']}/ValidateGeneration/innfer_ValidateGeneration_{cfg['name']}_{file_name}{val_name_ext}.sh'",
                "",
              ]

          if not args.step == "SnakeMakePost":
            # Plot synthetic vs simulated comparison
            if ind == 0 and file_name != "combined" and len(info["row"]) == 1:
              val.data_key = "full"
              val.PlotGenerationSummary(parameters[file_name]["unique_Y_values"], columns=info["columns"], extra_dir="GenerationSummary")
              val.data_key = "val"
              exit()
            val.PlotGeneration(info["row"], columns=info["columns"], extra_dir="GenerationTrue1D")
            val.PlotGeneration(info["row"], columns=info["columns"], extra_dir="GenerationTrue1DTransformed", transform=True)
            if len(val.X_columns) > 1 and not args.skip_generation_correlation:
              val.Plot2DPulls(info["row"], columns=info["columns"], extra_dir="GenerationTrue2DPulls")
              val.Plot2DUnrolledGeneration(info["row"], columns=info["columns"], extra_dir="GenerationTrue2D")
              val.Plot2DUnrolledGeneration(info["row"], columns=info["columns"], extra_dir="GenerationTrue2DTransformed", transform=True)
              val.PlotCorrelationMatrix(info["row"], columns=info["columns"], extra_dir="GenerationCorrelation")
            # Write a dummy for snakemake
            MakeDirectories(dummy_file_name)
            with open(dummy_file_name, 'w') as f: pass

        if args.step in ["ValidateInference", "Infer", "SnakeMakePost"]:

          # Skip if no Y columns to recover and not data
          if args.step in ["Infer","SnakeMakePost"] and not args.data_type == "data":
            if len(info["row"]) == 0: 
              continue

          # Build the likelihood
          if args.submit is None and not args.step == "SnakeMakePost":
            val.BuildLikelihood()

          # Run the debug step
          if args.sub_step == "Debug":
            val.DoDebug(info["row"], columns=info["columns"], resample=("unbinned" in args.likelihood_type))

          if args.step in ["ValidateInference","SnakeMakePost"] and not args.likelihood_type in ["binned_extended","binned"]:
            print("- Running validation inference")
            dummy_file_name = f"data/{cfg['name']}/{file_name}/ValidateInference/{file_name}{val_name_ext}.txt"

            # Submit to batch if not InitialFits (this is done later)
            if (args.submit is not None or args.step == "SnakeMakePost") and not args.sub_step in ["InitialFits","Summary"]:
              if not args.step == "SnakeMakePost":
                cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind}"
                options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/{args.sub_step}/innfer_{args.step}_{args.sub_step}_{cfg['name']}_{file_name}{val_name_ext}.sh", "sge_queue":args.sge_queue}
                from batch import Batch
                sub = Batch(options=options)
                sub.Run()
                continue
              else:
                cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --specific-val-ind={ind} --step=ValidateInference --sub-step=Collect" 
                options = {"job_name": f"jobs/{cfg['name']}/ValidateInference/Collect/innfer_ValidateInference_Collect_{cfg['name']}_{file_name}{val_name_ext}.sh"}
                from batch import Batch
                sub = Batch(options=options)
                sub._CreateBatchJob([cmd])
                collect_inputs = []
                collect_outputs = []
                for ind_for_col, col in enumerate(info["columns"]):
                  collect_outputs.append(f"    'data/{cfg['name']}/{file_name}/ValidateInference/bootstrap_results_{col}{val_name_ext}.yaml'")
                  if ind_for_col < len(info["columns"]) - 1:
                    collect_outputs[-1] += ","
                  for resampling_seed_for_collect in range(args.number_of_bootstraps):
                    bootstrap_name_ext_for_collect = f"_bootstrap_{resampling_seed_for_collect}"
                    collect_inputs.append(f"    'data/{cfg['name']}/{file_name}/ValidateInference/best_fit{val_name_ext}{bootstrap_name_ext_for_collect}.yaml'")
                    if resampling_seed_for_collect < args.number_of_bootstraps - 1:
                      collect_inputs[-1] += ","
                rules += [
                  f"rule ValidateInference_Collect_{file_name}{val_name_ext}:",
                  "  input:"
                ]
                rules += collect_inputs
                rules += [
                  "  output:"
                ]
                rules += collect_outputs
                rules += [
                  "  threads: 2",
                  "  params:",
                  "    runtime=3600,",
                  "    has_avx=1",
                  "  shell:",
                  f"    'bash jobs/{cfg['name']}/ValidateInference/Collect/innfer_ValidateInference_Collect_{cfg['name']}_{file_name}{val_name_ext}.sh'",
                  "",
                ]
                cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --specific-val-ind={ind} --step=ValidateInference --sub-step=Plot" 
                options = {"job_name": f"jobs/{cfg['name']}/ValidateInference/Plot/innfer_ValidateInference_Plot_{cfg['name']}_{file_name}{val_name_ext}.sh"}
                sub = Batch(options=options)
                sub._CreateBatchJob([cmd])
                rules += [
                  f"rule ValidateInference_Plot_{file_name}{val_name_ext}:",
                  "  input:"]
                rules += collect_outputs
                rules += [
                  "  output:",
                  f"    '{dummy_file_name}'",
                  "  threads: 2",
                  "  params:",
                  "    runtime=3600,",
                  "    has_avx=1",
                  "  shell:",
                  f"    'bash jobs/{cfg['name']}/ValidateInference/Plot/innfer_ValidateInference_Plot_{cfg['name']}_{file_name}{val_name_ext}.sh'",
                  "",
                ]

            # Run the bootstrapped maximum likelihood fits
            if args.sub_step in ["InitialFits","All"]:

              print("- Running the initial fits")

              # Loop through bootstrap resampling seeds
              for resampling_seed in range(args.number_of_bootstraps):

                # Setup name extensions
                bootstrap_name_ext = f"_bootstrap_{resampling_seed}"

                # Submit
                if args.submit is not None or args.step == "SnakeMakePost":
                  if not args.step == "SnakeMakePost":
                    cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--specific-bootstrap-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind} --specific-bootstrap-ind={resampling_seed}"
                    options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/{args.sub_step}/innfer_{args.step}_{args.sub_step}_{cfg['name']}_{file_name}{val_name_ext}{bootstrap_name_ext}.sh", "sge_queue":args.sge_queue}
                    from batch import Batch
                    sub = Batch(options=options)
                    sub.Run()
                    continue
                  else:
                    cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i  and '--specific-bootstrap-ind' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --specific-val-ind={ind} --specific-bootstrap-ind={resampling_seed} --step=ValidateInference --sub-step=InitialFits" 
                    options = {"job_name": f"jobs/{cfg['name']}/ValidateInference/InitialFits/innfer_ValidateInference_InitialFits_{cfg['name']}_{file_name}{val_name_ext}{bootstrap_name_ext}.sh"}
                    from batch import Batch
                    sub = Batch(options=options)
                    sub._CreateBatchJob([cmd])
                    rules += [
                      f"rule ValidateInference_InitialFits_{file_name}{val_name_ext}{bootstrap_name_ext}:",
                      "  input:"
                    ]
                    rules += model_inputs
                    rules += [
                      "  output:",
                      f"    'data/{cfg['name']}/{file_name}/ValidateInference/best_fit{val_name_ext}{bootstrap_name_ext}.yaml'",
                      "  threads: 2",
                      "  params:",
                      "    runtime=10800,",
                      "    has_avx=1",
                      "  shell:",
                      f"    'bash jobs/{cfg['name']}/ValidateInference/InitialFits/innfer_ValidateInference_InitialFits_{cfg['name']}_{file_name}{val_name_ext}{bootstrap_name_ext}.sh'",
                      "",
                    ]   

                # Run
                if (args.specific_bootstrap_ind is None or (str(resampling_seed) == args.specific_bootstrap_ind)) and not args.step == "SnakeMakePost":
                  val.GetAndDumpBestFit(info["initial_best_fit_guess"], row=info["row"], ind=ind,  columns=info["columns"], resampling_seed=resampling_seed, sampling_seed=resampling_seed, add_bootstrap=True, minimisation_method=args.minimisation_method, freeze=freeze)


            # Collect the bootstrapped fits
            if args.sub_step in ["Collect","All"] and not args.step == "SnakeMakePost":

              print("- Collecting bootstrapped fits")

              for col in info["columns"]:
                best_fit_info = {"all":{}}
                for resampling_seed in range(args.number_of_bootstraps):
                  bootstrap_name_ext = f"_bootstrap_{resampling_seed}"
                  with open(f"data/{cfg['name']}/{file_name}/{args.step}/best_fit{val_name_ext}{bootstrap_name_ext}.yaml", 'r') as yaml_file:
                    load_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    best_fit_info["all"][resampling_seed] = load_dict["best_fit"][load_dict["columns"].index(col)]
                best_fit_info["mean"] = float(np.mean([v for _,v in best_fit_info["all"].items()]))
                best_fit_info["std"] = float(np.std([v for _,v in best_fit_info["all"].items()]))
                # Dump to yaml
                filename = f"data/{cfg['name']}/{file_name}/{args.step}/bootstrap_results_{col}{val_name_ext}.yaml"
                print(f">> Created {filename}")
                with open(filename, 'w') as yaml_file:
                  yaml.dump(best_fit_info, yaml_file, default_flow_style=False)

            # Plot the bootstapped fits
            if args.sub_step in ["Plot","All"] and not args.step == "SnakeMakePost":

              print("- Plotting bootstrapped fits")

              combined_means = []
              for col in info["columns"]: 

                # Load in best fit information
                with open(f"data/{cfg['name']}/{file_name}/{args.step}/bootstrap_results_{col}{val_name_ext}.yaml", 'r') as yaml_file:
                  best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    
                # Draw best fit and interval distributions
                val.PlotBootstrappedDistribution(
                  [v for _, v in best_fit_info["all"].items()], 
                  col, 
                  info["row"],
                  info["columns"],
                  bins=20,
                  extra_dir="Bootstrapping"
                )
                combined_means.append(best_fit_info["mean"])

              # Draw comparisons between bestfit mean and truth
              val.PlotComparisons(info["row"], combined_means, columns=info["columns"], extra_dir="Comparisons")
              #val.PlotComparisonsSeparatingDatasets(info["row"], combined_means, columns=info["columns"], extra_dir="ComparisonsSeparatingDatasets")
              MakeDirectories(dummy_file_name)
              with open(dummy_file_name, 'w') as f: pass

          # Infer
          if args.step in ["Infer","SnakeMakePost"]:
        
            print("- Running inference")
            
            dummy_file_name = f"data/{cfg['name']}/{file_name}/Infer/{file_name}{val_name_ext}.txt"
            # Submit to batch
            if (args.submit is not None or args.step == "SnakeMakePost"):
              if not args.step in ["SnakeMakePost","Scan"]:
                cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind}"
                options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/{args.sub_step}/innfer_{args.step}_{args.sub_step}_{cfg['name']}_{file_name}{val_name_ext}.sh", "sge_queue":args.sge_queue}
                from batch import Batch
                sub = Batch(options=options)
                sub.Run()
                continue
              else:
                cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --specific-val-ind={ind} --step=Infer --sub-step=InitialFit" 
                options = {"job_name": f"jobs/{cfg['name']}/Infer/InitialFit/innfer_Infer_InitialFit_{cfg['name']}_{file_name}{val_name_ext}.sh"}
                from batch import Batch
                sub = Batch(options=options)
                sub._CreateBatchJob([cmd])
                rules += [
                  f"rule Innfer_InitialFit_{file_name}{val_name_ext}:",
                  "  input:"
                ]
                if not args.likelihood_type in ["binned","binned_extended"]:
                  rules += model_inputs
                else:
                  param_inputs = []
                  if args.specific_file == "combined":
                    for file_name_for_params in cfg["files"].keys():
                      param_inputs.append(f"    'data/{cfg['name']}/{file_name_for_params}/PreProcess/parameters.yaml',")
                    param_inputs[-1] = param_inputs[-1][:-1]
                  else:
                    param_inputs.append(f"    'data/{cfg['name']}/{file_name}/PreProcess/parameters.yaml'")
                scan_values_inputs = [f"    'data/{cfg['name']}/{file_name}/Infer/scan_values_{col}{val_name_ext}.yaml'," for col in info["columns"]]
                scan_values_inputs[-1] = scan_values_inputs[-1][:-1]
                rules += [
                  "  output:",
                  f"    'data/{cfg['name']}/{file_name}/Infer/best_fit{val_name_ext}.yaml',"
                ]
                rules += scan_values_inputs
                rules += [
                  "  threads: 1",
                  "  params:",
                  "    runtime=36000,",
                  "    gpu=1",
                  "  shell:",
                  f"    'bash jobs/{cfg['name']}/Infer/InitialFit/innfer_Infer_InitialFit_{cfg['name']}_{file_name}{val_name_ext}.sh'",
                  "",
                ]

                scan_outputs = []
                for col in info["columns"]:
                  job_ind = 0
                  scan_names = []
                  for point_ind in range(args.number_of_scan_points):
            
                    scan_name = f"    'data/{cfg['name']}/{file_name}/Infer/scan_results_{col}{val_name_ext}_scan_{point_ind}.yaml',"
                    scan_names.append(scan_name)
                    scan_outputs.append(scan_name)

                    if ((point_ind+1) % args.scan_points_per_job == 0) or (point_ind == args.number_of_scan_points-1):
                      scan_names[-1] = scan_names[-1][:-1]   
                      cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--specific-scan-ind' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --specific-val-ind={ind} --specific-scan-ind={job_ind} --step=Infer --sub-step=Scan" 
                      options = {"job_name": f"jobs/{cfg['name']}/Infer/Scan/innfer_Infer_Scan_{cfg['name']}_{file_name}{val_name_ext}_scan_{job_ind}.sh"}
                      from batch import Batch
                      sub = Batch(options=options)
                      sub._CreateBatchJob([cmd])
                      rules += [
                        f"rule Infer_Scan_{file_name}{val_name_ext}_scan_{job_ind}:",
                        "  input:",
                        f"    'data/{cfg['name']}/{file_name}/Infer/best_fit{val_name_ext}.yaml',",
                        f"    'data/{cfg['name']}/{file_name}/Infer/scan_values_{col}{val_name_ext}.yaml'",
                        "  output:"
                      ]
                      rules += scan_names
                      rules += [
                        "  threads: 2",
                        "  params:",
                        "    runtime=3600,",
                        "    has_avx=1",
                        "  shell:",
                        f"    'bash jobs/{cfg['name']}/Infer/Scan/innfer_Infer_Scan_{cfg['name']}_{file_name}{val_name_ext}_scan_{job_ind}.sh'",
                        "",
                      ]
                      job_ind += 1
                      scan_names = []

                scan_outputs[-1] = scan_outputs[-1][:-1]   

                cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --specific-val-ind={ind} --step=Infer --sub-step=Collect" 
                options = {"job_name": f"jobs/{cfg['name']}/Infer/Collect/innfer_Infer_Collect_{cfg['name']}_{file_name}{val_name_ext}.sh"}
                from batch import Batch
                sub = Batch(options=options)
                sub._CreateBatchJob([cmd])
                rules += [
                  f"rule Infer_Collect_{file_name}{val_name_ext}:",
                  "  input:"
                ]
                rules += scan_outputs
                rules += [
                  "  output:",
                  f"    'data/{cfg['name']}/{file_name}/Infer/scan_results_{col}{val_name_ext}.yaml'",
                  "  threads: 2",
                  "  params:",
                  "    runtime=3600,",
                  "    has_avx=1",
                  "  shell:",
                  f"    'bash jobs/{cfg['name']}/Infer/Collect/innfer_Infer_Collect_{cfg['name']}_{file_name}{val_name_ext}.sh'",
                  "",
                ]

                cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --specific-val-ind={ind} --step=Infer --sub-step=Plot" 
                options = {"job_name": f"jobs/{cfg['name']}/Infer/Plot/innfer_Infer_Plot_{cfg['name']}_{file_name}{val_name_ext}.sh"}
                from batch import Batch
                sub = Batch(options=options)
                sub._CreateBatchJob([cmd])
                rules += [
                  f"rule Infer_Plot_{file_name}{val_name_ext}:",
                  "  input:",
                  f"    'data/{cfg['name']}/{file_name}/Infer/scan_results_{col}{val_name_ext}.yaml'",
                  "  output:",
                  f"    '{dummy_file_name}'",
                  "  threads: 2",
                  "  params:",
                  "    runtime=3600,",
                  "    has_avx=1",
                  "  shell:",
                  f"    'bash jobs/{cfg['name']}/Infer/Plot/innfer_Infer_Plot_{cfg['name']}_{file_name}{val_name_ext}.sh'",
                  "",
                ]

            if not args.step == "SnakeMakePost":

              # Run the inital fits
              if args.sub_step in ["InitialFit","All"]:

                print("- Running initial fit")
                # Get the best fit
                val.GetAndDumpBestFit(info["initial_best_fit_guess"], row=info["row"], ind=ind,  columns=info["columns"], minimisation_method=args.minimisation_method, resample=False, freeze=freeze, scale_to_n_eff=args.scale_to_n_eff)
                # Get scan ranges
                for col in info["columns"]:
                  if not col in freeze.keys():
                    val.GetAndDumpScanRanges(col, row=info["row"], ind=ind, columns=info["columns"], resample=False, number_of_scan_points=args.number_of_scan_points, sigma_between_scan_points=args.sigma_between_scan_points, scale_to_n_eff=args.scale_to_n_eff)


              # Run the likelihood scans
              if args.sub_step in ["Scan","All"]:

                print("- Running scans")

                # Load best fit
                with open(f"data/{cfg['name']}/{file_name}/{args.step}/best_fit{val_name_ext}.yaml", 'r') as yaml_file:
                  best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

                if args.submit is None:
                  val.lkld.best_fit = np.array(best_fit_info["best_fit"])
                  val.lkld.best_fit_nll = best_fit_info["best_fit_nll"]

                # Loop through Y variables to get a scan for each Y value with the others profiled
                job_ind = 0

                for col in info["columns"]:

                  if col in freeze.keys():
                    continue

                  # Load scan ranges
                  with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_values_{col}{val_name_ext}.yaml", 'r') as yaml_file:
                    scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader) 

                  # Run scans
                  for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):

                    if not (args.specific_scan_ind != None and args.specific_scan_ind != str(job_ind)):
                      if args.submit is None:
                        val.GetAndDumpScan(info["row"], col, scan_value, ind1=ind, ind2=point_ind,  columns=info["columns"], minimisation_method=args.minimisation_method, resample=False, freeze=freeze, scale_to_n_eff=args.scale_to_n_eff)

                    if ((point_ind+1) % args.scan_points_per_job == 0) or (point_ind == len(scan_values_info["scan_values"])-1):
                      # Submit to batch
                      if args.submit is not None:
                        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--specific-scan-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind} --specific-scan-ind={job_ind}"
                        options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/{args.sub_step}/innfer_{args.step}_{args.sub_step}_{cfg['name']}_{file_name}{val_name_ext}_scan_{job_ind}.sh", "sge_queue":args.sge_queue}
                        from batch import Batch
                        sub = Batch(options=options)
                        sub.Run()
                      job_ind += 1

              if args.sub_step in ["Collect","All"]:

                print("- Collecting scans")

                for col in info["columns"]:

                  if col in freeze.keys():
                    continue

                  # Load scan ranges
                  with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_values_{col}{val_name_ext}.yaml", 'r') as yaml_file:
                    scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader) 

                  # Load scan results
                  scan_results = {"nlls":[],"scan_values":[]}
                  for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):      
                    with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}{val_name_ext}_scan_{point_ind}.yaml", 'r') as yaml_file:
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
                  filename = f"data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}{val_name_ext}.yaml"
                  print(f">> Created {filename}")
                  with open(filename, 'w') as yaml_file:
                    yaml.dump(scan_results, yaml_file, default_flow_style=False)
                  ## Remove per point yaml files
                  #for point_ind, scan_value in enumerate(scan_values_info["scan_values"]): 
                  #  os.system(f"rm data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}{val_name_ext}_scan_{point_ind}.yaml")

              if args.sub_step in ["Plot","All"]:

                print("- Plotting scans and comparisons")

                with open(f"data/{cfg['name']}/{file_name}/{args.step}/best_fit{val_name_ext}.yaml", 'r') as yaml_file:
                  best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                for col in info["columns"]:

                  if col in freeze.keys():
                    continue

                  with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}{val_name_ext}.yaml", 'r') as yaml_file:
                    scan_results_info= yaml.load(yaml_file, Loader=yaml.FullLoader)

                  val.PlotLikelihood(
                    scan_results_info["scan_values"], 
                    scan_results_info["nlls"], 
                    scan_results_info["row"], 
                    col, 
                    scan_results_info["crossings"], 
                    best_fit_info["best_fit"], 
                    #true_pdf=pdf, 
                    columns=info["columns"],
                    extra_dir="LikelihoodScans",
                    do_eff_weights=(args.likelihood_type in ["unbinned","unbinned_extended"])
                  )

                if not args.likelihood_type in ["binned_extended","binned"]:
                  val.PlotComparisons(best_fit_info["row"], best_fit_info["best_fit"], columns=info["columns"], extra_dir="Comparisons")
                  #val.PlotComparisonsSeparatingDatasets(best_fit_info["row"], best_fit_info["best_fit"], columns=info["columns"], extra_dir="ComparisonsSeparatingDatasets")
                  val.PlotGeneration(info["row"], columns=info["columns"], sample_row=best_fit_info["best_fit"], extra_dir="GenerationBestFit1D")
                  if len(val.X_columns) > 1:
                    val.Plot2DUnrolledGeneration(info["row"], columns=info["columns"], sample_row=best_fit_info["best_fit"], extra_dir="GenerationBestFit2D")
                else:
                  val.PlotBinned(info["row"], columns=info["columns"], sample_row=best_fit_info["best_fit"], extra_dir="BinnedDistributions")

                # Write a dummy for snakemake
                MakeDirectories(dummy_file_name)
                with open(dummy_file_name, 'w') as f: pass

      if args.sub_step in ["Summary","All"] and (args.submit is None or args.step == "SnakeMakePost") and args.specific_val_ind is None:

        if len(info["row"]) == 0: 
          continue

        validate_inference_summary_dummy_file_name = f"data/{cfg['name']}/{file_name}/ValidateInference/summary_{file_name}.txt"
        infer_summary_dummy_file_name = f"data/{cfg['name']}/{file_name}/Infer/summary_{file_name}.txt"

        if (args.submit is not None or args.step == "SnakeMakePost"):
          if not args.step == "SnakeMakePost":
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i])} --specific-file={file_name}"
            options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/{cfg['name']}/{args.step}/{args.sub_step}/innfer_{args.step}_{args.sub_step}_{cfg['name']}_{file_name}.sh", "sge_queue":args.sge_queue}
            from batch import Batch
            sub = Batch(options=options)
            sub.Run()
            continue
          else:
            if not args.likelihood_type in ["binned","binned_extended"]:
              cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --step=ValidateInference --sub-step=Summary" 
              options = {"job_name": f"jobs/{cfg['name']}/ValidateInference/Summary/innfer_ValidateInference_Summary_{cfg['name']}_{file_name}.sh"}
              from batch import Batch
              sub = Batch(options=options)
              sub._CreateBatchJob([cmd])
              rules += [
                f"rule ValidateInference_Summary_{file_name}:",
                "  input:"
              ]
              validate_inference_summary_inputs = []
              for ind, info in enumerate(val_loop):
                for col in info["columns"]:
                  if col in freeze.keys():
                    continue
                  validate_inference_summary_inputs.append(f"    'data/{cfg['name']}/{file_name}/ValidateInference/bootstrap_results_{col}_val_{ind}.yaml',")
              validate_inference_summary_inputs[-1] = validate_inference_summary_inputs[-1][:-1]
              rules += validate_inference_summary_inputs
              rules += [
                "  output:",
                f"    '{validate_inference_summary_dummy_file_name}'",
                "  threads: 2",
                "  params:",
                "    runtime=3600,",
                "    has_avx=1",
                "  shell:",
                f"    'bash jobs/{cfg['name']}/ValidateInference/Summary/innfer_ValidateInference_Summary_{cfg['name']}_{file_name}.sh'",
              ]
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--step' not in i and '--sub-step' not in i])} --specific-file={file_name} --step=Infer --sub-step=Summary" 
            options = {"job_name": f"jobs/{cfg['name']}/Infer/Summary/innfer_Infer_Summary_{cfg['name']}_{file_name}.sh"}
            from batch import Batch
            sub = Batch(options=options)
            sub._CreateBatchJob([cmd])
            rules += [
              f"rule Infer_Summary_{file_name}:",
              "  input:"
            ]
            innfer_summary_inputs = []
            for ind, info in enumerate(val_loop):
              for col in info["columns"]:
                if col in freeze.keys():
                  continue
                innfer_summary_inputs.append(f"    'data/{cfg['name']}/{file_name}/Infer/scan_results_{col}_val_{ind}.yaml',")
            innfer_summary_inputs[-1] = innfer_summary_inputs[-1][:-1]
            rules += innfer_summary_inputs
            rules += [
              "  output:",
              f"    '{infer_summary_dummy_file_name}'",
              "  threads: 2",
              "  params:",
              "    runtime=3600,",
              "    has_avx=1",
              "  shell:",
              f"    'bash jobs/{cfg['name']}/Infer/Summary/innfer_Infer_Summary_{cfg['name']}_{file_name}.sh'",
            ]


        if args.step == "ValidateInference" and not args.likelihood_type in ["binned","binned_extended"]:

          print("- Plotting summary of the validation using inference")

          results = {}
          for ind, info in enumerate(val_loop):
            info_name = GetYName(info["row"],purpose="file")
            plot_name = GetYName(info["row"],purpose="plot",prefix="y=")
            for col in info["columns"]:
              if col in freeze.keys():
                continue
              with open(f"data/{cfg['name']}/{file_name}/{args.step}/bootstrap_results_{col}_val_{ind}.yaml", 'r') as yaml_file:
                best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
              intervals = {
                  -2 : best_fit_info["mean"] - 2*best_fit_info["std"],
                  -1 : best_fit_info["mean"] - 1*best_fit_info["std"],
                  0 : best_fit_info["mean"],
                  1 : best_fit_info["mean"] + 1*best_fit_info["std"],
                  2 : best_fit_info["mean"] + 2*best_fit_info["std"],
                }
              norm_intervals = {k:v/info["row"][info["columns"].index(col)] for k,v in intervals.items()}
              if col not in results.keys():
                results[col] = {plot_name:norm_intervals}
              else:
                results[col][plot_name] = norm_intervals

          val.PlotSummary(results, extra_dir="Summary")

          # Write a dummy for snakemake
          MakeDirectories(validate_inference_summary_dummy_file_name)
          with open(validate_inference_summary_dummy_file_name, 'w') as f: pass

        elif args.step == "Infer":

          print("- Plotting summary of the results")

          results = {}
          for ind, info in enumerate(val_loop):
            info_name = GetYName(info["row"],purpose="file")
            for col in info["columns"]:
              if col in freeze.keys():
                continue
              with open(f"data/{cfg['name']}/{file_name}/{args.step}/scan_results_{col}_val_{ind}.yaml", 'r') as yaml_file:
                scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
              if col not in results.keys():
                results[col] = {info_name:scan_results_info}
              else:
                results[col][info_name] = scan_results_info

          # Plot summary
          #val.PlotValidationSummary(results, true_pdf=pdf, extra_dir="Summary")
          val.PlotInferSummary(results, extra_dir="Summary")

          # Plot prediction summaries
          if args.data_type == "asimov":
            print()

          # Write a dummy for snakemake
          MakeDirectories(infer_summary_dummy_file_name)
          with open(infer_summary_dummy_file_name, 'w') as f: pass

  if args.step in ["MakePDFs"]:
    
    import glob
    from plotting import make_pdf_summary
    list_plots = {
      "PreProcess_X_distributions_$KEY_train" : "PreProcess/X_distributions_$KEY_against_*_train.pdf",
      "PreProcess_X_distributions_$KEY_train_transformed" : "PreProcess/X_distributions_$KEY_against_*_train_transformed.pdf",
      "PreProcess_X_distributions_$KEY_test" : "PreProcess/X_distributions_$KEY_against_*_test.pdf",
      "PreProcess_X_distributions_$KEY_test_transformed" : "PreProcess/X_distributions_$KEY_against_*_test_transformed.pdf",
      "PreProcess_X_distributions_$KEY_val" : "PreProcess/X_distributions_$KEY_against_*_val.pdf",
      "PreProcess_X_distributions_$KEY_val_transformed" : "PreProcess/X_distributions_$KEY_against_*_val_transformed.pdf",
      "PreProcess_Y_distributions" : "PreProcess/Y_distributions_*.pdf",
      "Train" : "Train*.pdf",
      "ValidateGeneration_$KEY_1D" : "ValidateGeneration/GenerationTrue1D/generation_*_sim_$KEY_synth_*.pdf",
      "ValidateGeneration_$KEY_1D_transformed" : "ValidateGeneration/GenerationTrue1DTransformed/generation_*_sim_$KEY_synth_*.pdf",
      "ValidateGeneration_1D" : "ValidateGeneration/GenerationTrue1D/generation_*.pdf",
      "ValidateGeneration_1D_transformed" : "ValidateGeneration/GenerationTrue1D/generation_*.pdf",
      "ValidateGeneration_$KEY_2D" : "ValidateGeneration/GenerationTrue2D/generation_unrolled_2d_*_sim_$KEY_*.pdf",
      "ValidateGeneration_$KEY_2D_transformed" : "ValidateGeneration/GenerationTrue2DTransformed/generation_unrolled_2d_*_sim_$KEY_*.pdf",
      "ValidateGeneration_$KEY_2D_pulls" : "ValidateGeneration/GenerationTrue2DPulls/generation_pulls_2d_*_sim_$KEY_*.pdf",
      "ValidateGeneration_2D" : "ValidateGeneration/GenerationTrue2D/generation_unrolled_2d_*.pdf",
      "ValidateGeneration_2D_transformed" : "ValidateGeneration/GenerationTrue2DTransformed/generation_unrolled_2d_*.pdf",
      "ValidateGeneration_2D_pulls" : "ValidateGeneration/GenerationTrue2DPulls/generation_pulls_2d_*.pdf",
      "ValidateInference_Bootstrapping" : "ValidateInference/Bootstrapping/*.pdf",
      "ValidateInference_Comparisons" : "ValidateInference/Comparisons/*.pdf",
      "ValidateInference_Summary" : "ValidateInference/Summary/*.pdf",
      "Infer_LikelihoodScans" : "Infer/LikelihoodScans/*.pdf",
      "Infer_Comparisons" : "Infer/Comparisons/*.pdf",
      "Infer_Generation_1D_$KEY" : "Infer/GenerationBestFit1D/generation_*_sim_$KEY_synth*.pdf",
      "Infer_Generation_2D_$KEY" : "Infer/GenerationBestFit2D/generation_*_sim_$KEY_synth*.pdf",
      "Infer_Summary" : "Infer/Summary/*.pdf",
    }

    for file_name in file_loop:
      for pdf_name, plots in list_plots.items():
        prename = f"plots/{cfg['name']}/{file_name}/"

        split = plots.split("$KEY")

        if len(split) == 1:

          files = glob.glob(prename+plots)
          output_name = f"plots/{cfg['name']}/{file_name}/Merged/{pdf_name}"
          MakeDirectories(output_name+".pdf")
          if len(files) == 0: continue
          make_pdf_summary(list(np.sort(files)), name=output_name)

        elif len(split) == 2:

          # find keys 
          name_with_wildcard = prename+plots.replace("$KEY","*")
          files = glob.glob(name_with_wildcard)
          wildcard_splits = name_with_wildcard.split("*")
          keys = []
          for f in files:
            wildcards = []
            pre_wildcard = ""
            for i in range(len(wildcard_splits)-1):
              if i != 0: pre_wildcard += wildcards[-1]
              pre_wildcard += wildcard_splits[i]
              wildcards.append(f.split(pre_wildcard)[1].split(wildcard_splits[i+1])[0])
            key = wildcards[plots.split("$KEY")[0].count("*")]
            if key not in keys: keys.append(key)
          if len(keys) == 0: continue

          # get files
          for key in keys:
            files = glob.glob(prename+plots.replace("$KEY",key))
            if len(files) == 0: continue
            output_name = f"plots/{cfg['name']}/{file_name}/Merged/{pdf_name.replace('$KEY',key)}"
            MakeDirectories(output_name+".pdf")
            make_pdf_summary(list(np.sort(files)), name=output_name)

        else:
          print(f"WARNING: Skipping {pdf_name} for {file_name} as $KEY not correctly defined.")

  if args.step in ["SnakeMakePre", "SnakeMakePost"]:

    rules_all = [
      "rule all:",
      "  input:",
    ]

    all_inputs = []
    is_output = False
    is_not_deleted = True
    for r in rules:
      #if "rule" in r and ("InitialFits" in r or "Scan" in r):
      #  is_not_deleted = False
      #elif "rule" in r:
      #  is_not_deleted = True
      if "output:" in r:
        is_output = True
        continue
      elif ":" in r:
        is_output = False
      if is_output and is_not_deleted:
        all_inputs.append(r.replace(",",""))

    for ind_for_all_input, all_input in enumerate(all_inputs):
      rules_all.append(all_input)
      if ind_for_all_input < len(all_inputs) - 1:
        rules_all[-1] += ","
      
    rules = rules_all + [""] + rules

    sm_add = ""
    if args.submit is not None:
      if not args.submit in ["condor","cern_condor"]:
        raise NotImplementedError(f"{args.submit} is not implemented yet with snakemake.")
      else:
        sm_add += " --profile htcondor"
    else:
      sm_add += " --jobs 10"

    sm_name = f"jobs/{cfg['name']}/{args.step}.txt"
    MakeDirectories(sm_name)
    with open(sm_name, "w") as file:
      for r in rules:
        file.write(str(r) + "\n")
    print(f"Created {sm_name}")
    os.system(f"snakemake -s {sm_name} --unlock") 
    os.system(f"snakemake -s {sm_name} {sm_add}") 

if __name__ == "__main__":

  start_time = time.time()
  title = pyg.figlet_format("INNFER")
  print()
  print(title)

  args = parse_args()

  if not args.step == "SnakeMake":
    main(args)
  else:
    args.step = "SnakeMakePre"
    main(args) # Run snakemake for before training
    args.step = "SnakeMakePost"
    main(args) # Run snakemake for training and after

  print("- Finished running without error")
  end_time = time.time()
  hours, remainder = divmod(end_time-start_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f"- Time elapsed: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")