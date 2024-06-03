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

def CustomHistogram(data, weights=None, bins=20, density=False, discrete_binning=True, add_uncert=False):
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
  if not add_uncert:
    hist, bins = np.histogram(data, weights=weights, bins=bins, density=density)
    return hist, bins
  else:
    hist, bins = np.histogram(data, weights=weights, bins=bins)
    hist_wt_squared, _ = np.histogram(data, weights=weights**2 if weights is not None else None, bins=bins)
    hist_uncert = np.sqrt(hist_wt_squared)
    if density:
      hist_uncert /= np.sum(hist)
      hist /= np.sum(hist)
    return hist, hist_uncert, bins   

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

def Resample(datasets, weights, n_samples=None, seed=42):

  # Set up datasets as lists
  if not isinstance(datasets, list):
    datasets = [datasets]

  # Set n_samples to the maximum if None
  if n_samples is None:
    n_samples = len(weights)

  # Get positive and negative weights indices
  positive_indices = (weights>=0)
  negative_indices = (weights<0)
  do_positive = False
  do_negative = False
  if np.sum(weights[positive_indices]) != 0:
    do_positive = True
  if np.sum(weights[negative_indices]) != 0:
    do_negative = True
  
  if do_positive:
    positive_probs = weights[positive_indices]/np.sum(weights[positive_indices])
  if do_negative:
    negative_probs = weights[negative_indices]/np.sum(weights[negative_indices])
  
  # Set n_samples to the maximum if None
  n_positive_samples = 0
  n_negative_samples = 0
  if n_samples is None:
    if do_positive:
      n_positive_samples = len(positive_probs)
    if do_negative:
      n_negative_samples = len(negative_probs)
  else:
    if do_positive and do_negative: 
      positive_fraction = len(positive_probs)/len(weights)
      n_positive_samples = int(round(positive_fraction*n_samples,0))
      n_negative_samples = n_samples - n_positive_samples
    elif do_positive:
      n_positive_samples = n_samples
    elif do_negative:
      n_negative_samples = n_samples

  # Loop through datasets
  resampled_datasets = []
  resampled_weights = np.hstack((np.ones(n_positive_samples),-1*np.ones(n_negative_samples)))
  for dataset in datasets:

    # Skip if empty
    if dataset.shape[1] == 0:
      resampled_datasets.append(np.zeros((n_positive_samples+n_negative_samples, 0)))
      continue

    # Resample
    rng = np.random.RandomState(seed=seed)
    if do_positive:
      positive_resampled_indices = rng.choice(len(dataset[positive_indices]), size=n_positive_samples, p=positive_probs)
    if do_negative:
      negative_resampled_indices = rng.choice(len(dataset[negative_indices]), size=n_negative_samples, p=negative_probs)

    # Save
    if do_positive and do_negative:
      resampled_datasets.append(np.vstack((dataset[positive_indices][positive_resampled_indices],dataset[negative_indices][negative_resampled_indices])))
    elif do_positive:
      resampled_datasets.append(dataset[positive_indices][positive_resampled_indices])
    elif do_negative:
      resampled_datasets.append(dataset[negative_indices][negative_resampled_indices])
    
  # Return individual dataset if originally given
  if len(resampled_datasets) == 1:
    resampled_datasets = resampled_datasets[0]

  return resampled_datasets, resampled_weights

def FindKeysAndValuesInDictionaries(config, keys=[], results_keys=[], results_vals=[]):
  """
  Find keys and values in dictionaries.

  Args:
      config (dict): Configuration dictionary.
      keys (list): List of keys (default is []).
      results_keys (list): List of results keys (default is []).
      results_vals (list): List of results values (default is []).

  Returns:
      tuple: Tuple containing lists of results keys and results values.
  """
  for k, v in config.items():
    new_keys = keys+[k]
    if isinstance(v, dict):
      results_keys, results_vals = FindKeysAndValuesInDictionaries(v, keys=new_keys, results_keys=results_keys, results_vals=results_vals)
    else:
      results_keys.append(new_keys)
      results_vals.append(v)
  return results_keys, results_vals


def MakeDictionaryEntry(dictionary, keys, val):
  """
  Make a dictionary entry.

  Args:
      dictionary (dict): Dictionary.
      keys (list): List of keys.
      val (object): Value.

  Returns:
      dict: Dictionary containing the entry.
  """
  if len(keys) > 1:
    if keys[0] not in dictionary.keys():
      dictionary[keys[0]] = {}
    dictionary[keys[0]] = MakeDictionaryEntry(dictionary[keys[0]], keys[1:], val)
  else:
    dictionary[keys[0]] = val
  return dictionary


def GetScanArchitectures(cfg, data_output="data/", write=True):
  """
  Get scan architectures.

  Args:
      config (dict): Configuration dictionary file.

  Returns:
      list: List of scan architectures.
  """

  # open config
  with open(cfg, 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)  

  keys, vals = FindKeysAndValuesInDictionaries(config, keys=[], results_keys=[], results_vals=[])
  all_lists = [v for v in vals if isinstance(v,list)]
  ind_lists = [ind for ind in range(len(vals)) if isinstance(vals[ind],list)]
  unique_vals = list(product(*all_lists))

  count = 0
  outputs = []
  for uv in unique_vals:

    # Set unique values
    ind_val = list(vals)
    for i in range(len(ind_lists)):
      ind_val[ind_lists[i]] = uv[i]

    output = {}
    for ind in range(len(keys)):
      output = MakeDictionaryEntry(output, keys[ind], ind_val[ind])

    name = f"{data_output}/scan_architecture_{count}.yaml"
    outputs.append(name)

    if write:
      MakeDirectories(name)
      with open(name, 'w') as yaml_file:
        yaml.dump(output, yaml_file, default_flow_style=False) 
    
    count += 1

  return outputs