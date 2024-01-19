import copy
import numpy as np
from itertools import product
from scipy.interpolate import RegularGridInterpolator


def GetValidateLoop(cfg, parameters_file):
  """
  Generate a list of dictionaries representing parameter combinations for validation loops.

  Args:
      cfg (dict): Configuration dictionary containing "pois" and "nuisances".
      parameters_file (dict): Dictionary containing parameter information.

  Returns:
      list: List of dictionaries representing parameter combinations for validation loops.
  """
  val_loop = []

  pois = [poi for poi in cfg["pois"] if poi in parameters_file["Y_columns"]]
  unique_values_for_pois = [parameters_file["unique_Y_values"][poi] for poi in pois]
  nuisances = [nuisance for nuisance in cfg["nuisances"] if nuisance in parameters_file["Y_columns"]]
  unique_values_for_nuisances = [parameters_file["unique_Y_values"][nuisance] for nuisance in nuisances]
  initial_poi_guess = [sum(parameters_file["unique_Y_values"][poi])/len(parameters_file["unique_Y_values"][poi]) for poi in pois]
  initial_best_fit_guess = np.array(initial_poi_guess+[0.0]*(len(nuisances)))        

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
  """
  Generate a list of dictionaries representing parameter combinations for POI loops.

  Args:
      cfg (dict): Configuration dictionary containing "pois" and "nuisances".
      parameters (dict): Dictionary containing parameter information.

  Returns:
      list: List of dictionaries representing parameter combinations for POI loops.
  """
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
  """
  Generate a list of dictionaries representing parameter combinations for nuisance loops.

  Args:
      cfg (dict): Configuration dictionary containing "pois" and "nuisances".
      parameters (dict): Dictionary containing parameter information.

  Returns:
      list: List of dictionaries representing parameter combinations for nuisance loops.
  """
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
  """
  Generate a list of dictionaries representing parameter combinations for combined validation loops.

  Args:
      cfg (dict): Configuration dictionary containing "pois", "nuisances", "inference", and "validation".
      parameters (dict): Dictionary containing parameter information.

  Returns:
      list: List of dictionaries representing parameter combinations for combined validation loops.
  """
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
    initial_poi_guess += [1.0]*len(pois)

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
      purpose (str): Purpose of the label, either "plot" or "file".
      round_to (int): Number of decimal places to round the values (default is 2).

  Returns:
      str: Formatted label for the given unique row.
  """
  if len(ur) > 0:
    label_list = [str(round(i,round_to)) for i in ur] 
    if purpose in ["file","code"]:
      name = "_".join([i.replace(".","p").replace("-","m") for i in label_list])
    elif purpose == "plot":
      if len(label_list) > 1:
        name = "({})".format(",".join(label_list))
      else:
        name = label_list[0]
  else:
    if purpose in ["file","plot"]:
      name = ""
    else:
      name = None
  return name

def GetVariedRowValue(poi_vars, nuisance_vars, parameters, poi, nuisance, nuisance_val, return_dict, entry_dict):

  # Find initial options
  #entry_dict = {}
  #for ind in range(len(parameters["Y_columns"])):
  #  entry_dict[parameters["Y_columns"][ind]] = sorted(list(set([float(k.split("_")[ind].replace("p",".").replace("m","-")) for k in parameters["yield"].keys()])))

  # Get name to add
  row_to_add = []
  for col in parameters["Y_columns"]:
    if col in poi_vars:
      row_to_add.append(poi[poi_vars.index(col)])
    elif col == nuisance:
      row_to_add.append(nuisance_val)
    elif col in nuisance_vars:
      row_to_add.append(0.0)        
  row_to_add_name = GetYName(row_to_add, purpose="file")

  # Skip if already in list
  if row_to_add_name in parameters["yield"]: 
    return return_dict

  # Figure out which dimension to vary
  missing_col_ind = 0
  for col_ind, col in enumerate(parameters["Y_columns"]):
    if not row_to_add[col_ind] in entry_dict[col]:
      missing_col_ind = copy.deepcopy(col_ind)

  # find closest other val
  lst = copy.deepcopy(entry_dict[parameters["Y_columns"][missing_col_ind]])
  if row_to_add[missing_col_ind] in lst: lst.remove(row_to_add[missing_col_ind])
  x1 = min(lst, key=lambda x: abs(x - row_to_add[missing_col_ind]))
  lst.remove(x1)
  x2 = min(lst, key=lambda x: abs(x - row_to_add[missing_col_ind]))
  row_1_for_line = copy.deepcopy(row_to_add)
  row_1_for_line[missing_col_ind] = x1
  row_2_for_line = copy.deepcopy(row_to_add)
  row_2_for_line[missing_col_ind] = x2

  #print(row_1_for_line, row_2_for_line, row_to_add)

  # Get straight line
  y1 = parameters["yield"][GetYName(row_1_for_line, purpose="file")]
  y2 = parameters["yield"][GetYName(row_2_for_line, purpose="file")]
  m = (y2 - y1) / (x2 - x1)
  c = y2 - (m * x2)
  y = (m * row_to_add[missing_col_ind]) + c
  return_dict[row_to_add_name] = y
  
  #print(row_1_for_line, row_2_for_line, row_to_add, y1, y2, y)

  return return_dict

def MakeYieldFunction(poi_vars, nuisance_vars, parameters, add_overflow=0.25):

  parameters = copy.deepcopy(parameters)

  if not "all" in parameters["yield"].keys():

    # Find initial options
    orig_dict = {}
    for ind in range(len(parameters["Y_columns"])):
      orig_dict[parameters["Y_columns"][ind]] = sorted(list(set([float(k.split("_")[ind].replace("p",".").replace("m","-")) for k in parameters["yield"].keys()])))

    # Extend range
    val_dict = copy.deepcopy(orig_dict)
    nuisance_loop = nuisance_vars if len(nuisance_vars) > 0 else [None]

    orig_dict[None] = [None]
    add_dict = {}
    if add_overflow > 0.0:
      for k, v in val_dict.items():
        col_range = max(v) - min(v)
        add_dict[k] = [min(v) - (add_overflow*col_range), max(v) + (add_overflow*col_range)]
        val_dict[k] = sorted(v + add_dict[k])
      unique_pois = [val_dict[poi] for poi in poi_vars if poi in val_dict.keys()]
      mesh_pois = list(product(*unique_pois))

      for poi in mesh_pois:
        for nuisance in nuisance_loop:
          if nuisance not in orig_dict.keys(): continue
          # Do inside the nuisance range
          for nuisance_val in orig_dict[nuisance]:
            parameters["yield"] = GetVariedRowValue(poi_vars, nuisance_vars, parameters, poi, nuisance, nuisance_val, parameters["yield"], orig_dict)

      for k, v in add_dict.items():
        if k in poi_vars:
          orig_dict[k] += v

      for poi in mesh_pois:
        for nuisance in nuisance_loop:
          if nuisance not in add_dict.keys(): continue
          # Do outside the nuisance range
          for nuisance_val in add_dict[nuisance]:
            parameters["yield"] = GetVariedRowValue(poi_vars, nuisance_vars, parameters, poi, nuisance, nuisance_val, parameters["yield"], orig_dict)

    # Get unique values and mesh
    unique_points = val_dict.values()
    mesh = [np.array(i).flatten() for i in np.meshgrid(*unique_points,indexing="ij")]

    #for k, v in parameters["yield"].items():
    #  print(k,v)
    #print(unique_points)
    #exit()

    # Get values
    data = []
    for ind in range(len(mesh[0])):
      row = [mesh[col][ind] for col in range(len(mesh))]
      row_name = GetYName(row, purpose="file")

      # If exact row exists use that
      if row_name in parameters["yield"].keys():

        data.append(parameters["yield"][row_name])

      # Else assume uncorrelated and add up changes from the nominal nuisance value
      else:

        zero_nuisance_row = []
        for col_ind, col in enumerate(parameters["Y_columns"]):
          if col in poi_vars:
            zero_nuisance_row.append(mesh[col_ind][ind])
          elif col in nuisance_vars:
            zero_nuisance_row.append(0.0)
        zero_nuisance_row_name = GetYName(zero_nuisance_row, purpose="file")
        zero_nuisance_val = parameters["yield"][zero_nuisance_row_name]

        zero_individual_nuisance_vals = []
        for nuisance in nuisance_loop:

          zero_individual_nuisance_row = []
          for col_ind, col in enumerate(parameters["Y_columns"]):
            if col != nuisance:
              zero_individual_nuisance_row.append(mesh[col_ind][ind])
            else:
              zero_individual_nuisance_row.append(0.0)
          zero_individual_nuisance_row_name = GetYName(zero_individual_nuisance_row, purpose="file")   
          zero_individual_nuisance_vals.append(parameters["yield"][zero_individual_nuisance_row_name])       

        sum_diff = sum([v-zero_nuisance_val for v in zero_individual_nuisance_vals])
        data.append(zero_nuisance_val+sum_diff)

    data_array_shape = tuple(len(uv) for uv in unique_points)
    data = np.array(data).reshape(data_array_shape)

    interp = RegularGridInterpolator(unique_points, data, method='linear', bounds_error=False, fill_value=np.nan)

    def func(x):
      return interp(x)[0]
    
  else:
    
    def func(x):
      return parameters["yield"]["all"]

  return func