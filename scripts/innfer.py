import argparse
import yaml
import os
import sys
from itertools import product
from other_functions import GetValidateLoop, GetPOILoop, GetNuisanceLoop

print("Running INNFER")

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cfg', help= 'Config for running',  default=None)
parser.add_argument('--benchmark', help= 'Run from benchmark scenario',  default=None, choices=["Gaussian","GaussianWithExpBkg"])
parser.add_argument('--architecture', help= 'Config for running',  default="configs/architecture/default.yaml")
parser.add_argument('--submit', help= 'Batch to submit to', type=str, default=None)
parser.add_argument('--step', help= 'Step to run', type=str, default=None, choices=["PreProcess","Train","Validate","Infer"])
parser.add_argument('--specific-file', help= 'Run for a specific file_name', type=str, default=None)
parser.add_argument('--specific-ind', help= 'Run for a specific indices when doing validation', type=str, default=None)
parser.add_argument('--disable-tqdm', help= 'Disable tqdm print out when training.',  action='store_true')
parser.add_argument('--sge-queue', help= 'Queue for SGE submission', type=str, default="hep.q")
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
  if args.step in ["Train","Validate","Infer"]:

    with open(f"data/{cfg['name']}/{file_name}/parameters.yaml", 'r') as yaml_file:
      parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if args.submit is None:

      if args.step != "Train":
        with open(f"models/{cfg['name']}/{file_name}_architecture.yaml", 'r') as yaml_file:
          architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

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
      networks[file_name].BuildModel()

      ### Train or load networks ###
      if args.step == "Train":
        print("- Training model")
        networks[file_name].disable_tqdm =  args.disable_tqdm
        networks[file_name].BuildTrainer()
        networks[file_name].Train()
        networks[file_name].Save(name=f"models/{cfg['name']}/{file_name}.h5")
        with open(f"models/{cfg['name']}/{file_name}_architecture.yaml", 'w') as file:
          yaml.dump(architecture, file)
      else:
        networks[file_name].Load(name=f"models/{cfg['name']}/{file_name}.h5")    
  
    ### Do validation of trained model ###
    if args.step == "Validate":
      print("- Performing validation")
      if args.submit is None:
        from validation import Validation
        val = Validation(
          networks[file_name], 
          options={
            "data_parameters":parameters[file_name],
            "data_dir":f"data/{cfg['name']}/{file_name}",
            "plot_dir":f"plots/{cfg['name']}/{file_name}",
            "model_name":file_name
            }
          )
        networks[file_name].data_parameters = parameters[file_name]

      for ind, info in enumerate(GetValidateLoop(cfg, parameters[file_name])):

        if args.specific_ind != None and args.specific_ind != str(ind): continue

        print(f" - Columns: {info['columns']}")
        print(f" - Values: {info['row']}")

        # Submit to batch
        if args.submit is not None:
          cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-ind'not in i])} --specific-file={file_name} --specific-ind={ind}"
          options = {"submit_to": args.submit, "cmds": [cmd], "job_name": f"jobs/innfer_{args.step.lower()}_{cfg['name']}_{file_name}_{ind}.sh", "sge_queue":args.sge_queue}
          from batch import Batch
          sub = Batch(options=options)
          sub.Run()
          continue

        # Plot synthetic vs simulated comparison
        val.PlotGeneration(info["row"], columns=info["columns"])
        # Plot unbinned likelihood closure and best fit learned distributions
        val.PlotUnbinnedLikelihood(info["row"], info["initial_best_fit_guess"], columns=info["columns"], true_pdf=(None if args.benchmark is None else benchmark.GetPDF))

  ### Do inference on data ###
  if args.step == "Infer":
    print("- Performing inference")