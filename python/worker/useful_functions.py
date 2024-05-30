import copy
import os
import yaml
import pandas as pd
import numpy as np
from itertools import product

def CheckRunAndSubmit(
  argv,
  submit = None,
  loop = {}, 
  specific = "",
  job_name = "job",
  dry_run = False
  ):

  # Add '' when appropriate to command
  corrected_argv = []
  for arg in argv:
    if arg.count("=") > 1:
      corrected_argv.append(arg.split("=")[0] + "='" + "=".join(arg.split("=")[1:]) + "'")
    else:
      corrected_argv.append(arg)
  cmd = " ".join(corrected_argv)

  # Set up specific dictionary
  if specific == "":
    specific_dict = {}
  else:
    specific_dict = {i.split("=")[0]:i.split("=")[1].split(",") for i in specific.split(";")}

  # If the loop val is not in the specific dictionary do not run
  run = True
  for k, v in specific_dict.items():
    if k in loop.keys():
      if str(loop[k]) not in v:
        run = False

  if not run:
    return False

  if run:

    # run if not submit
    if submit is None:
      return True

    # set up batch submission command
    specific_str = None
    sub_str = None
    for string in corrected_argv:
      if string.startswith("--specific="):
        specific_str = copy.deepcopy(string)
      if string.startswith("--submit="):
        sub_str = copy.deepcopy(string)
    if specific_str is not None: corrected_argv.remove(specific_str)
    if sub_str is not None: corrected_argv.remove(sub_str)
    specific_sub_string = ";".join([f"{k}={v}" for k, v in loop.items()])
    if specific_sub_string != "":
      specific_sub_string = f" --specific='{specific_sub_string}'"
    sub_cmd = f"python3 {' '.join(corrected_argv)} {specific_sub_string}"  

    # Make extra name
    extra_name = "_".join([f"{k}_{v}" for k, v in loop.items()])
    if extra_name != "":
      extra_name = f"_{extra_name}"

    # Open config
    with open(submit, 'r') as yaml_file:
      submit_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Submit job
    options = {"submit_to": submit_cfg["name"], "options": submit_cfg["options"],"cmds": [sub_cmd], "job_name": StringToFile(f"{job_name}{extra_name}.sh"), "dry_run": dry_run}
    from batch import Batch
    sub = Batch(options=options)
    sub.Run()

    return False

def StringToFile(string):
  string = string.replace(",","_").replace(";","_").replace(">=","_geq_").replace("<=","_leq_").replace(">","_g_").replace("<","_l_").replace("!=","_noteq_").replace("=","_eq_")
  if string.count(".") > 1:
    string = f'{"p".join(string.split(".")[:-1])}.{string.split(".")[-1]}'
  return string

def MakeDirectories(file_loc):
  """
  Make directories.

  Args:
      file_loc (str): File location.

  Returns:
      None
  """
  if file_loc[0] == "/":
    initial = "/"
    file_loc = file_loc[1:]
  else:
    initial = ""

  splitting = file_loc.split("/")
  for ind in range(len(splitting)):

    # Skip if file
    if "." in splitting[ind]: continue

    # Get full directory
    full_dir = initial + "/".join(splitting[:ind+1])

    # Make directory if it is missing
    if not os.path.isdir(full_dir): 
      os.system(f"mkdir {full_dir}")

def CustomHistogram(data, weights=None, bins=20, density=False, discrete_binning=True):
  """
  Compute a custom histogram for the given data.

  Args:
      data (numpy.ndarray): Input data.
      weights (numpy.ndarray, optional): Weights associated with the data. Defaults to None.
      bins (int or array_like, optional): If bins is an integer, it defines the number of equal-width
          bins in the range. If bins is an array, it defines the bin edges. Defaults to 20.

  Returns:
      numpy.ndarray: Histogram of data.
      numpy.ndarray: Bin edges.
  """
  unique_vals = pd.DataFrame(data).drop_duplicates()
  if isinstance(bins, int):
    if len(unique_vals) < bins and discrete_binning:
      bins = np.sort(unique_vals.to_numpy().flatten())
      bins = np.append(bins, [(2*bins[-1]) - bins[-2]])
  hist, bins = np.histogram(data, weights=weights, bins=bins, density=density)
  return hist, bins

def GetYName(ur, purpose="plot", round_to=2, prefix=""):
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
    label_list = [str(round(float(i),round_to)) for i in ur] 
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

  if name is not None and name != "":
    name = prefix+name
  return name

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
    nuisance_freeze={k:0.0 for k in cfg["nuisances"] if k != nuisance}
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