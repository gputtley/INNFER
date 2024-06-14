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

def CustomHistogram(data, weights=None, bins=20, density=False, discrete_binning=True, add_uncert=False, ignore_quantile=0.0):
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
  
  unique_vals = data.drop_duplicates()

  # Discrete values for bins
  db = False
  if isinstance(bins, int):

    if len(unique_vals) < bins and discrete_binning:
      db = True
      bins = np.sort(unique_vals.to_numpy().flatten())
      bins = np.append(bins, [(2*bins[-1]) - bins[-2]])

    elif ignore_quantile > 0.0:
      # Ignore quantiles
      qdown = data.quantile(ignore_quantile)
      qup = data.quantile(1-ignore_quantile)
      trimmed_indices = data[data.between(qdown, qup)].index
      data = data.loc[trimmed_indices]
      if weights is not None:
        weights = weights.loc[trimmed_indices]

  # run without uncertainties
  if not add_uncert:
    hist, bins = np.histogram(data, weights=weights, bins=bins, density=density)
    return hist, bins

  # run with uncertainties
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
  if isinstance(ur, pd.DataFrame):
    ur = ur.to_numpy().flatten()

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
  #initial_poi_guess = [sum(parameters_file["unique_Y_values"][poi])/len(parameters_file["unique_Y_values"][poi]) for poi in pois]
  average_pois = [sum(parameters_file["unique_Y_values"][poi])/len(parameters_file["unique_Y_values"][poi]) for poi in pois]
  initial_poi_guess = [min(unique_values_for_pois[ind], key=lambda x: abs(x - average_pois[ind])) for ind in range(len(pois))]
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
        sorted_columns = sorted(columns)
        sorted_row = [row[columns.index(col)] for col in sorted_columns]
        sorted_initial_best_fit_guess = [initial_best_fit_guess[columns.index(col)] for col in sorted_columns]
        val_loop.append({
          "row" : pd.DataFrame([sorted_row], columns=sorted_columns),
          "initial_best_fit_guess" : pd.DataFrame([sorted_initial_best_fit_guess], columns=sorted_columns),
        })
        
  return val_loop

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
    average_pois = [sum(uv)/len(uv) for uv in unique_values_for_pois]
    initial_poi_guess += [min(unique_values_for_pois[ind], key=lambda x: abs(x - average_pois[ind])) for ind in range(len(average_pois))]

  for poi in cfg["pois"]:
    pois.append(poi)
    unique_values_for_pois.append([])
    for file in cfg["files"]:
      if poi not in parameters[file]["unique_Y_values"]: continue
      for val in parameters[file]["unique_Y_values"][poi]:
        if val not in unique_values_for_pois[-1]:
          unique_values_for_pois[-1].append(val)

    average_poi = sum(unique_values_for_pois[-1])/len(unique_values_for_pois[-1])
    initial_poi_guess.append(min(unique_values_for_pois[-1], key=lambda x: abs(x - average_poi)))

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
        sorted_columns = sorted(columns)
        sorted_row = [row[columns.index(col)] for col in sorted_columns]
        sorted_initial_best_fit_guess = [initial_best_fit_guess[columns.index(col)] for col in sorted_columns]
        val_loop.append({
          "row" : pd.DataFrame([sorted_row], columns=sorted_columns),
          "initial_best_fit_guess" : pd.DataFrame([sorted_initial_best_fit_guess], columns=sorted_columns),
        })

  return val_loop


def GetValidateInfo(preprocess_loc, models_loc, cfg):
  parameters = {file_name: f"{preprocess_loc}/{file_name}/PreProcess/parameters.yaml" for file_name in cfg["files"].keys()}
  loaded_parameters = {file_name: yaml.load(open(file_loc), Loader=yaml.FullLoader) for file_name, file_loc in parameters.items()}
  val_loops = {file_name: GetValidateLoop(cfg, loaded_parameters[file_name]) for file_name in cfg["files"].keys()}
  models = {file_name: f"{models_loc}/{file_name}/{file_name}.h5" for file_name in cfg["files"].keys()}
  architectures = {file_name: f"{models_loc}/{file_name}/{file_name}_architecture.yaml" for file_name in cfg["files"].keys()}
  if len(cfg["files"].keys()) > 1:
    val_loops["combined"] = GetCombinedValidateLoop(cfg, loaded_parameters)
    parameters["combined"] = copy.deepcopy(parameters)
    models["combined"] = copy.deepcopy(models)
    architectures["combined"] = copy.deepcopy(architectures)
  info = {
    "val_loops" : val_loops,
    "parameters" : parameters,
    "models" : models,
    "architectures" : architectures,
  }
  return info

def FindEqualStatBins(data, bins=5, sf_diff=2):
    diff = data.quantile(0.75) - data.quantile(0.25)
    significant_figures = sf_diff - int(np.floor(np.log10(abs(diff)))) - 1
    rounded_number = round(diff, significant_figures)
    decimal_places = len(str(rounded_number).rstrip('0').split(".")[1])
    
    unique_vals = data.drop_duplicates()

    # Discrete and less than bins
    if len(unique_vals) < bins:
      equal_bins = list(np.sort(unique_vals.to_numpy().flatten())[1:])
    # Discrete
    elif len(unique_vals) < 20:
      sorted_bins = np.sort(unique_vals.to_numpy().flatten())
      equal_bins = [round(data.quantile(i/bins), min(decimal_places,significant_figures)) for i in range(1,bins)]
      equal_bins = list(set(equal_bins))
      if equal_bins[0] == sorted_bins[0]:
        equal_bins = equal_bins[1:]
    # Continuous
    else:
      equal_bins = [round(data.quantile(i/bins), min(decimal_places,significant_figures)) for i in range(1,bins)]

    equal_bins = [-np.inf] + equal_bins + [np.inf]

    return equal_bins

def RoundToSF(num, sig_figs):
  if num == 0:
    return 0
  else:
    rounded_number = round(num, sig_figs)
    decimal_places = len(str(rounded_number).rstrip('0').split(".")[1])
    return round(num, decimal_places)


def SetupSnakeMakeFile(args, default_args, main):

  # Define variables
  clear_file = True
  rules_all = [
    "rule all:",
    "  input:",
  ]
  all_lines = []
  output_line = False

  # Open snakemake config
  with open(args.snakemake_cfg, 'r') as yaml_file:
    snakemake_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

  # Make snakemake file
  args.make_snakemake_inputs = True
  for step, step_info in snakemake_cfg.items():
    args.step = step
    if "submit" in step_info.keys():
      args.submit = step_info["submit"]
    if "run_options" in step_info.keys():
      for arg_name, arg_val in step_info["run_options"].items():
        setattr(args, arg_name, arg_val)

    # Open config when available
    if args.cfg is not None and clear_file:
      with open(args.cfg, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
      snakemake_file = f"jobs/{cfg['name']}/innfer_SnakeMake.txt"
      if os.path.isfile(snakemake_file):
        os.system(f"rm {snakemake_file}")
      clear_file = False

    # Write info to file
    main(args, default_args)

  # make final outputs
  with open(args.cfg, 'r') as yaml_file:
    cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
  snakemake_file = f"jobs/{cfg['name']}/innfer_SnakeMake.txt"

  # Find outputs
  with open(snakemake_file, 'r') as file:
    for line in file:
      line = line.rstrip()
      all_lines.append(line)
      if line.startswith("  shell:") or line.startswith("  input:") or line.startswith("  params:") or line == "":
        output_line = False
        continue
      elif line.startswith("  output:"):
        output_line = True
        continue
      if output_line:
        rules_all.append(line)

  rules = rules_all+[""]+all_lines
  from batch import Batch
  b_rules = Batch()
  b_rules._CreateJob(rules, snakemake_file)   

  # Run snakemake
