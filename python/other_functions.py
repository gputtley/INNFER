import copy
import numpy as np
from itertools import product

def GetValidateLoop(cfg, parameters_file):
  val_loop = []

  if len(parameters_file["Y_columns"]) > 0:

    pois = [poi for poi in cfg["pois"] if poi in parameters_file["Y_columns"]]
    unique_values_for_pois = [parameters_file["unique_Y_values"][poi] for poi in pois]
    nuisances = [nuisance for nuisance in cfg["nuisances"] if nuisance in parameters_file["Y_columns"]]
    unique_values_for_nuisances = [parameters_file["unique_Y_values"][nuisance] for nuisance in nuisances]
    initial_poi_guess = [sum(parameters_file["unique_Y_values"][poi])/len(parameters_file["unique_Y_values"][poi]) for poi in pois]
    initial_best_fit_guess = np.array(initial_poi_guess+[0]*(len(nuisances)))        

    for poi_values in list(product(*unique_values_for_pois)): 
      nuisances_loop = [None] if len(nuisances) == 0 else copy.deepcopy(nuisances)
      for ind, nuisance in enumerate(nuisances_loop):
        nuisance_value_loop = [None] if len(nuisances) == 0 else unique_values_for_nuisances[ind]
        for nuisance_value in nuisance_value_loop:
          other_nuisances = [v for v in cfg["nuisances"] if v != nuisance]
          nuisance_in_row = [] if nuisance is None else [nuisance]
          nuisance_value_in_row = [] if nuisance_value is None else [nuisance_value]
          row = np.array(list(poi_values)+nuisance_value_in_row+[0]*len(other_nuisances))
          columns = pois+nuisance_in_row+other_nuisances
          val_loop.append({
            "row" : row,
            "columns" : columns,
            "initial_best_fit_guess" : initial_best_fit_guess,
          })
        
  return val_loop


def GetPOILoop(cfg, parameters):
  poi_loop = []

  for poi in cfg["pois"]:
    nuisance_freeze = {k:0 for k in cfg["nuisances"]}
    other_pois = [v for v in cfg["pois"] if v != poi and v in parameters["unique_Y_values"]]
    unique_values_for_other_pois = [parameters["unique_Y_values"][other_poi] for other_poi in other_pois]
    for other_poi_values in list(product(*unique_values_for_other_pois)):
      poi_freeze = {other_pois[ind]: val for ind, val in enumerate(other_poi_values)}
      if len(poi_freeze.keys())>0:
        poi_freeze_name = "_for_" + "_".join([f"{k}_eq_{str(v).replace('.','p')}" for k, v in poi_freeze.items()])
      else:
        poi_freeze_name = ""
      poi_loop.append({
        "poi" : poi, 
        "freeze" : {**nuisance_freeze, **poi_freeze}, 
        "extra_name" : poi_freeze_name
      })

  return poi_loop

def GetNuisanceLoop(cfg, parameters):
  nuisance_loop = []

  for nuisance in cfg["nuisances"]:
    nuisance_freeze={k:0 for k in cfg["nuisances"] if k != nuisance}
    pois = [v for v in cfg["pois"] if v in parameters["unique_Y_values"]]
    unique_values_for_pois = [parameters["unique_Y_values"][poi] for poi in pois]
    for poi_values in list(product(*unique_values_for_pois)):
      poi_freeze = {pois[ind]: val for ind, val in enumerate(poi_values)}
      if len(poi_freeze.keys())>0:
        poi_freeze_name = "_for_" + "_".join([f"{k}_eq_{str(v).replace('.','p')}" for k, v in poi_freeze.items()])
      else:
        poi_freeze_name = ""
      nuisance_loop.append({
        "nuisance" : nuisance, 
        "freeze" : {**nuisance_freeze, **poi_freeze}, 
        "extra_name" : poi_freeze_name
      })

  return nuisance_loop

def GetCombinedValidateLoop(cfg, parameters):
  val_loop = []

  pois = []
  unique_values_for_pois = []
  nuisances = []
  unique_values_for_nuisances = []
  initial_poi_guess = []
  initial_best_fit_guess = []

  if "rate_parameters" in cfg["inference"]:
    pois += [f"mu_{rp}" for rp in cfg["inference"]["rate_parameters"]]
    unique_values_for_pois += [cfg["validation"]["rate_parameter_vals"][rp] for rp in cfg["inference"]["rate_parameters"]]
    initial_poi_guess += [0.0]*len(pois)

  for poi in cfg["pois"]:
    pois.append(poi)
    unique_values_for_pois.append([])
    for file in cfg["files"]:
      if poi not in parameters[file]["unique_Y_values"]: continue
      for val in parameters[file]["unique_Y_values"][poi]:
        if val not in unique_values_for_pois[-1]:
          unique_values_for_pois[-1].append(val)
    initial_poi_guess.append(sum(unique_values_for_pois[-1])/len(unique_values_for_pois[-1]))

  for nuisance in cfg["nuisances"]:
    nuisances.append(nuisance)
    unique_values_for_nuisances.append([0.0])

  initial_best_fit_guess = np.array(initial_poi_guess+[0]*(len(nuisances)))

  for poi_values in list(product(*unique_values_for_pois)): 
    nuisances_loop = [None] if len(nuisances) == 0 else copy.deepcopy(nuisances)
    for ind, nuisance in enumerate(nuisances_loop):
      nuisance_value_loop = [None] if len(nuisances) == 0 else unique_values_for_nuisances[ind]
      for nuisance_value in nuisance_value_loop:
        other_nuisances = [v for v in cfg["nuisances"] if v != nuisance]
        nuisance_in_row = [] if nuisance is None else [nuisance]
        nuisance_value_in_row = [] if nuisance_value is None else [nuisance_value]
        row = np.array(list(poi_values)+nuisance_value_in_row+[0]*len(other_nuisances))
        columns = pois+nuisance_in_row+other_nuisances
        val_loop.append({
          "row" : list(row),
          "columns" : list(columns),
          "initial_best_fit_guess" : list(initial_best_fit_guess),
        })

  return val_loop

def GetYName(ur, purpose="plot", round_to=2):
  """
  Get a formatted label for a given unique row.

  Args:
      ur (list): List representing the unique row.
      purpose (str): Purpose of the label, either "plot" or "file
  """
  if len(ur) > 0:
    label_list = [str(round(i,round_to)) for i in ur] 
    if purpose == "file":
      name = "_".join([i.replace(".","p").replace("-","m") for i in label_list])
    elif purpose == "plot":
      if len(label_list) > 1:
        name = "({})".format(",".join(label_list))
      else:
        name = label_list[0]
  else:
    name = ""
  return name