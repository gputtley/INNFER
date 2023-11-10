import argparse
import yaml
import os
import pandas as pd
import numpy as np
from itertools import product
from preprocess import PreProcess
from network import Network
from validation import Validation

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
for file_name, _ in cfg["files"].items():
  if not os.path.isdir(f"data/{cfg['name']}/{file_name}"): os.system(f"mkdir data/{cfg['name']}/{file_name}")
  if not os.path.isdir(f"models/{cfg['name']}/{file_name}"): os.system(f"mkdir models/{cfg['name']}/{file_name}")
  if not os.path.isdir(f"plots/{cfg['name']}/{file_name}"): os.system(f"mkdir plots/{cfg['name']}/{file_name}")  

if cfg["preprocess"]["standardise"] == "all":
  cfg["preprocess"]["standardise"] = cfg["variables"] + cfg["pois"]+cfg["nuisances"]

networks = {}
parameters = {}
pp = {}

for file_name, parquet_name in cfg["files"].items():

  if args.specific != None and args.specific != file_name: continue

  ### PreProcess Data ###
  if args.step == "PreProcess":
    print("- Preprocessing data")

    # PreProcess the dataset
    pp[file_name] = PreProcess(parquet_name, cfg["variables"], cfg["pois"]+cfg["nuisances"], options=cfg["preprocess"])
    pp[file_name].output_dir = f"data/{cfg['name']}/{file_name}"
    pp[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}"
    pp[file_name].Run()

    # Run plots varying the pois across the variables
    for poi in cfg["pois"]:
      nuisance_freeze = {k:0 for k in cfg["nuisances"]}
      other_pois = [v for v in cfg["pois"] if v != poi and v in pp[file_name].parameters["unique_Y_values"]]
      unique_values_for_other_pois = [pp[file_name].parameters["unique_Y_values"][other_poi] for other_poi in other_pois]
      for other_poi_values in list(product(*unique_values_for_other_pois)):
        poi_freeze = {other_pois[ind]: val for ind, val in enumerate(other_poi_values)}
        if len(poi_freeze.keys())>0:
          poi_freeze_name = "_for_" + "_".join([f"{k}_eq_{str(v).replace('.','p')}" for k, v in poi_freeze.items()])
        else:
          poi_freeze_name = ""
        pp[file_name].PlotX(poi, freeze={**nuisance_freeze, **poi_freeze}, dataset="train", extra_name=poi_freeze_name)

    # Run plots varying the nuisances across the variables for each unique value of the pois
    for nuisance in cfg["nuisances"]:
      nuisance_freeze={k:0 for k in cfg["nuisances"] if k != nuisance}
      pois = [v for v in cfg["pois"] if v in pp[file_name].parameters["unique_Y_values"]]
      unique_values_for_pois = [pp[file_name].parameters["unique_Y_values"][poi] for poi in pois]
      for poi_values in list(product(*unique_values_for_pois)):
        poi_freeze = {pois[ind]: val for ind, val in enumerate(poi_values)}
        if len(poi_freeze.keys())>0:
          poi_freeze_name = "_for_" + "_".join([f"{k}_eq_{str(v).replace('.','p')}" for k, v in poi_freeze.items()])
        else:
          poi_freeze_name = ""
        pp[file_name].PlotX(nuisance, freeze={**nuisance_freeze, **poi_freeze}, dataset="train", extra_name=poi_freeze_name)

    # Run plots of the distribution of the context features
    pp[file_name].PlotY(dataset="train")


  ### Training, validation and inference ####
  if args.step in ["Train","Validation","Inference"]:

    with open(f"data/{cfg['name']}/{file_name}/parameters.yaml", 'r') as yaml_file:
      parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

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
      networks[file_name].BuildTrainer()
      networks[file_name].Train()
      networks[file_name].Save(name=f"models/{cfg['name']}/{file_name}.h5")
    else:
      networks[file_name].Load(name=f"models/{cfg['name']}/{file_name}.h5")    
  
    ### Do validation of trained model ###
    if args.step == "Validation":
      print("- Performing validation")
      val = Validation(networks[file_name], options={"data_parameters":parameters[file_name],"data_dir":f"data/{cfg['name']}/{file_name}"})
      val.plot_dir = f"plots/{cfg['name']}/{file_name}"
      networks[file_name].data_parameters = parameters[file_name]

      # Make generation plots
      pois = [poi for poi in cfg["pois"] if poi in parameters[file_name]["Y_columns"]]
      unique_values_for_pois = [parameters[file_name]["unique_Y_values"][poi] for poi in pois]
      nuisances = [nuisance for nuisance in cfg["nuisances"] if nuisance in parameters[file_name]["Y_columns"]]
      unique_values_for_nuisances = [parameters[file_name]["unique_Y_values"][nuisance] for nuisance in nuisances]
      for poi_values in list(product(*unique_values_for_pois)):
        for ind, nuisance in enumerate(nuisances):
          nuisance_values = unique_values_for_nuisances[ind]
          for nuisance_value in nuisance_values:
            other_nuisances = [v for v in cfg["nuisances"] if v != nuisance]
            row = np.array(list(poi_values)+[nuisance_value]+[0]*len(other_nuisances))
            columns = pois+[nuisance]+other_nuisances
            val.PlotGeneration(row, columns=columns)

  ### Do inference on data ###
  if args.step == "Inference":
    print("- Performing inference")