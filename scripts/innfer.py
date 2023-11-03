import argparse
import yaml
import os
from preprocess import PreProcess
from network import Network

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cfg', help= 'Config for running',  default=None)
parser.add_argument('--architecture', help= 'Config for running',  default="configs/architecture/default.yaml")
parser.add_argument('--submit', help= 'Batch to submit to', type=str, default=None)
parser.add_argument('--step', help= 'Step to run', type=str, default=None, choices=["PreProcess","Train","Validation","Inference"])
parser.add_argument('--specific', help= 'Run for a specific file_name', type=str, default=None)
args = parser.parse_args()

if args.cfg is None:
  raise ValueError("The --cfg is required.")
if args.step is None:
  raise ValueError("The --step is required.")

with open(args.cfg, 'r') as yaml_file:
  cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

with open(args.architecture, 'r') as yaml_file:
  architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

if not os.path.isdir(f"data/{cfg['name']}"): os.system(f"mkdir data/{cfg['name']}")
if not os.path.isdir(f"models/{cfg['name']}"): os.system(f"mkdir models/{cfg['name']}")
if not os.path.isdir(f"plots/{cfg['name']}"): os.system(f"mkdir plots/{cfg['name']}")

if cfg["preprocess"]["standardise"] == "all":
  cfg["preprocess"]["standardise"] = cfg["variables"] + cfg["pois"]+cfg["nuisances"]

### PreProcess Data ###

if args.step == "PreProcess":
  print("- Preprocessing data")
  pp = {}
  for file_name, parquet_name in cfg["files"].items():

    if args.specific != None and args.specific != file_name: continue

    pp[file_name] = PreProcess(parquet_name, cfg["variables"], cfg["pois"]+cfg["nuisances"], options=cfg["preprocess"])
    pp[file_name].output_dir = f"data/{cfg['name']}/{file_name}"
    pp[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}"
    pp[file_name].Run()

    for poi in cfg["pois"]:
      nuisance_freeze={k:0 for k in cfg["nuisances"]}
      poi_freeze = {k:"central" for k in cfg["pois"] if k != poi}
      pp[file_name].PlotX([poi], freeze={**nuisance_freeze, **poi_freeze}, dataset="train")

    for nuisance in cfg["nuisances"]:
      nuisance_freeze={k:0 for k in cfg["nuisances"] if k != nuisance}
      poi_freeze = {k:"central" for k in cfg["pois"]}
      pp[file_name].PlotX([nuisance], freeze={**nuisance_freeze, **poi_freeze}, dataset="train")

    pp[file_name].PlotY(dataset="train")

### Build and train or load networks ###

networks = {}
for file_name, parquet_name in cfg["files"].items():

  if args.specific != None and args.specific != file_name: continue

  networks[file_name] = Network(
    f"data/{cfg['name']}/{file_name}_X_train.parquet",
    f"data/{cfg['name']}/{file_name}_Y_train.parquet", 
    f"data/{cfg['name']}/{file_name}_wt_train.parquet", 
    f"data/{cfg['name']}/{file_name}_X_test.parquet",
    f"data/{cfg['name']}/{file_name}_Y_test.parquet", 
    f"data/{cfg['name']}/{file_name}_wt_test.parquet",
    options=architecture)
  
  if args.step == "Train":
    print("- Training model")
    networks[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}"
    networks[file_name].BuildModel()
    networks[file_name].BuildTrainer()
    networks[file_name].Train()
    networks[file_name].Save(name=f"data/{cfg['name']}/{file_name}.h5")
  else:
    print("- Loading weights")
    networks[file_name].Load(name=f"data/{cfg['name']}/{file_name}.h5")

### Do validation of trained model ###

if args.step == "Validation":
  print("- Performing validation")

### Do inference on data ###

if args.step == "Inference":
  print("- Performing inference")