import copy
import os
import pickle
import re
import yaml

import numpy as np
import pandas as pd

from itertools import product

def BuildBinnedCategories(var_input):

  # Make selection, variable and bins
  categories = {}
  for category_ind, category in enumerate(var_input.split(";")):

    # Make selection, variable and bins
    if ":" in category:
      sel = category.split(":")[0]
      var_and_bins = category.split(":")[1]
    else:
      sel = None
      var_and_bins = category

    if "[" in var_and_bins:
      var = var_and_bins.split("[")[0]
      bins = [float(i) for i in list(var_and_bins.split("[")[1].split("]")[0].split(","))]
    elif "(" in var_and_bins:
      var = var_and_bins.split("(")[0]
      start_stop_step = [float(i) for i in list(var_and_bins.split("(")[1].split(")")[0].split(","))]
      bins = [float(i) for i in np.arange(start_stop_step[0], start_stop_step[1], start_stop_step[2])]
    categories[category_ind] = [sel, var, bins]
  
  return categories


def CamelToSnake(name):
  """
  Convert a CamelCase string to a snake_case string.

  Parameters
  ----------
  name : str
      The CamelCase string to be converted.

  Returns
  -------
  str
      The converted snake_case string.
  """
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
  return s2.lower()


def CommonInferConfigOptions(args, cfg, val_info, file_name, val_ind):

  defaults_in_model = GetDefaultsInModel(file_name, cfg, include_rate=args.include_per_model_rate, include_lnN=args.include_per_model_lnN)

  data_dir = str(os.getenv("DATA_DIR"))
  plots_dir = str(os.getenv("PLOTS_DIR"))
  models_dir = str(os.getenv("MODELS_DIR"))

  if args.data_type == "sim":
    data_input = {k:[f"{data_dir}/{cfg['name']}/PreProcess/{k}/val_ind_{v}/{i}_{args.sim_type}.parquet" for i in ["X","wt"]] for k,v in GetCombinedValdidationIndices(cfg, file_name, val_ind).items()}
  elif args.data_type == "asimov":
    data_input = {k:[f"{data_dir}/{cfg['name']}/MakeAsimov/{k}/val_ind_{v}/asimov.parquet"] for k,v in GetCombinedValdidationIndices(cfg, file_name, val_ind).items()}
  elif args.data_type == "data":
    data_input = {"data" : [cfg["data_file"]] if isinstance(cfg["data_file"], str) else cfg["data_file"]}

  if args.likelihood_type == "binned":
    print("Not implemented yet")

  common_config = {
    "density_models" : {k:GetModelLoop(cfg, model_file_name=k, only_density=True)[0] for k in ([file_name] if file_name != "combined" else GetModelFileLoop(cfg))},
    "regression_models" : {k:GetModelLoop(cfg, model_file_name=k, only_regression=True) for k in ([file_name] if file_name != "combined" else GetModelFileLoop(cfg))},
    "model_input" : f"{models_dir}/{cfg['name']}",
    "extra_density_model_name" : args.extra_density_model_name,
    "parameters" : {k:f"{data_dir}/{cfg['name']}/PreProcess/{k}/parameters.yaml" for k in GetCombinedValdidationIndices(cfg, file_name, val_ind).keys()},
    "data_input" : data_input,
    "true_Y" : pd.DataFrame({k: [v] if k not in val_info.keys() else [val_info[k]] for k, v in defaults_in_model.items()}),
    "initial_best_fit_guess" : pd.DataFrame({k:[v] for k, v in defaults_in_model.items()}),
    "inference_options" : cfg["inference"] if not args.no_constraint else {k: v for k, v in cfg["inference"].items() if k != 'nuisance_constraints'},
    "likelihood_type": args.likelihood_type,
    "scale_to_eff_events": args.scale_to_eff_events,
    "verbose": not args.quiet,
    "data_file": cfg["data_file"],
    "binned_fit_input" : args.binned_fit_input,
    "minimisation_method" : args.minimisation_method,
    "sim_type" : args.sim_type,
    "X_columns" : cfg["variables"],
    "Y_columns" : list(defaults_in_model.keys()),
    "Y_columns_per_model" : {k: GetParametersInModel(k, cfg) for k in ([file_name] if file_name != "combined" else GetModelFileLoop(cfg))},
    "only_density" : args.only_density,
    "non_nn_columns" : [k for k in defaults_in_model.keys() if k in list(cfg["inference"]["lnN"].keys()) or k.startswith("mu_")],
    #"binned_fit_data_input" : 
  }

  return common_config


def CustomHistogram(
    data, 
    weights = None, 
    bins = 20, 
    density = False, 
    discrete_binning = True, 
    add_uncert = False, 
    ignore_quantile = 0.0
  ):
  """
  Compute a custom histogram for the given data.

  Parameters
  ----------
  data : numpy.ndarray
      Input data.
  weights : numpy.ndarray, optional
      Weights associated with the data. Defaults to None.
  bins : int or array_like, optional
      If bins is an integer, it defines the number of equal-width bins in the range.
      If bins is an array, it defines the bin edges. Defaults to 20.
  density : bool, optional
      If True, the result is the value of the probability density function at the bin,
      normalized such that the integral over the range is 1. Defaults to False.
  discrete_binning : bool, optional
      If True, uses discrete binning. Defaults to True.
  add_uncert : bool, optional
      If True, calculates and returns uncertainties. Defaults to False.
  ignore_quantile : float, optional
      If greater than 0.0, ignores the given quantile range. Defaults to 0.0.

  Returns
  -------
  numpy.ndarray
      Histogram of data.
  numpy.ndarray
      Bin edges.
  numpy.ndarray, optional
      Uncertainties (if add_uncert is True).
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


def DiscreteTransform(df, splines={}, thresholds={}, n_integral_bins=100000, X_columns=[], Y_columns=[], wt_name="wt", unique_y_vals={}):

  unique_combinations = list(product(*unique_y_vals.values()))

  for uv in unique_combinations:

    for col in df.columns:

      if col not in thresholds.keys(): 
        continue

      for k, v in thresholds[col].items():

        # Find unique values
        unique_selections = {key: uv[ind] for ind, key in enumerate(unique_y_vals.keys())}
        indices = pd.Series([True] * len(df))
        for uv_col, uv_val in unique_selections.items():
          indices = indices & (df.loc[:,uv_col] == uv_val)

        # Find matching indices
        indices = indices & (df.loc[:,col] == k)
        n_samples = len(df.loc[indices,col]) 

        # Load spline
        with open(splines[col], 'rb') as file:
          spline = pickle.load(file)

        # Build flat sampling distribution
        wts = df.loc[indices,wt_name].to_numpy()
        sorted_indices = np.argsort(wts)
        uniform = np.linspace(v[0], v[1], num=n_samples)

        sorted_indices = np.argsort(uniform)
        unsorted_indices = np.arange(0,n_samples)[sorted_indices]
        uniform = uniform[sorted_indices]
        wts = wts[sorted_indices]

        cdf = np.cumsum(wts)
        cdf = cdf / cdf[-1]
        sampled = np.interp(uniform, uniform, cdf)

        sampled = sampled[unsorted_indices]

        # Compute the CDF
        param_values = np.linspace(v[0], v[1], n_integral_bins, )
        cdf_vals = np.cumsum(np.abs(spline(param_values))) / np.sum(np.abs(spline(param_values)))

        # Normalise the CDF
        cdf_vals /= cdf_vals[-1]

        # Inverse transform sampling
        df.loc[indices,col] = np.interp(sampled, cdf_vals, param_values)

  return df.loc[:,X_columns]


def FindEqualStatBins(data, bins=5, sf_diff=2):
  """
  Find bins with equal statistical weight for the given data.

  Parameters
  ----------
  data : pandas.Series
      Input data.
  bins : int, optional
      Number of bins. Defaults to 5.
  sf_diff : int, optional
      Significant figures difference for rounding. Defaults to 2.

  Returns
  -------
  list
      Bin edges with equal statistical weight.
  """

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


def FindKeysAndValuesInDictionaries(config, keys=[], results_keys=[], results_vals=[]):
  """
  Find keys and values in dictionaries.

  Parameters
  ----------
  config : dict
      Configuration dictionary.
  keys : list, optional
      List of keys. Defaults to [].
  results_keys : list, optional
      List of results keys. Defaults to [].
  results_vals : list, optional
      List of results values. Defaults to [].

  Returns
  -------
  tuple
      Tuple containing lists of results keys and results values.
  """

  for k, v in config.items():
    new_keys = keys+[k]
    if isinstance(v, dict):
      results_keys, results_vals = FindKeysAndValuesInDictionaries(v, keys=new_keys, results_keys=results_keys, results_vals=results_vals)
    else:
      results_keys.append(new_keys)
      results_vals.append(v)
  return results_keys, results_vals


def GetDictionaryEntry(entry, keys):
  """
  Retrieve a nested dictionary entry.

  Parameters
  ----------
  entry : dict
      The dictionary to search in.
  keys : list of str
      A list of keys to navigate through the nested dictionary.

  Returns
  -------
  object or None
      The value associated with the provided keys if the file and keys exist,
      otherwise None.
  """

  for key in keys:
    if isinstance(entry, dict):
      if key not in entry.keys():
        return None
    if isinstance(entry, list):
      if key >= len(entry):
        return None    
    entry = entry[key]
  return entry


def GetDictionaryEntryFromYaml(file_name, keys):
  """
  Retrieve a nested dictionary entry from a YAML file using a list of keys.

  Parameters
  ----------
  file_name : str
      The path to the YAML file.
  keys : list of str
      A list of keys to navigate through the nested dictionary.

  Returns
  -------
  object or None
      The value associated with the provided keys if the file and keys exist,
      otherwise None.
  """

  if os.path.isfile(file_name):
    with open(file_name, 'r') as yaml_file: 
      entry = yaml.load(yaml_file, Loader=yaml.FullLoader)
  else:
    return None

  for key in keys:
    if isinstance(entry, dict):
      if key not in entry.keys():
        return None
    if isinstance(entry, list):
      if key >= len(entry):
        return None    
    entry = entry[key]
  return entry


def GetBestFitFromYaml(file_name):

  if not os.path.isfile(file_name):
    return None

  with open(file_name, 'r') as yaml_file:
    entry = yaml.load(yaml_file, Loader=yaml.FullLoader)

  return {entry["columns"][ind]: entry["best_fit"][ind] for ind in range(len(entry["columns"]))}


def GetBaseFileLoop(cfg):
  return list(cfg["files"].keys())


def GetModelFileLoop(cfg, with_combined=False):
  model_files = list(cfg["models"].keys())
  if with_combined and len(model_files) > 1: 
    model_files += ["combined"]
  return model_files


def GetModelLoop(cfg, only_density=False, only_regression=False, model_file_name=None):

  data_dir = str(os.getenv("DATA_DIR"))
  plots_dir = str(os.getenv("PLOTS_DIR"))
  models_dir = str(os.getenv("MODELS_DIR"))

  models = []
  for k, v in cfg["models"].items():

    if model_file_name is not None:
      if model_file_name != k:
        continue

    if not only_regression:
      models.append(
        {
          "type" : "density",
          "file_loc" : f"{data_dir}/{cfg['name']}/PreProcess/{k}/density",
          "name" : f"density_{k}",
          "parameters" : f"{data_dir}/{cfg['name']}/PreProcess/{k}/parameters.yaml",
          "parameter" : None,
          "file_name" : k,
        }
      )
    for value in v["regression_models"]:
      if not only_density:
        models.append(
          {
            "type" : "regression",
            "file_loc" : f"{data_dir}/{cfg['name']}/PreProcess/{k}/regression/{value['parameter']}",
            "name" : f"regression_{k}_{value['parameter']}",
            "parameters" : f"{data_dir}/{cfg['name']}/PreProcess/{k}/parameters.yaml",
            "parameter" : value['parameter'],
            "file_name" : k,
          }
        )
  return models


def GetDefaults(cfg):
  defaults = {}
  for param in cfg["pois"]:
    if param in cfg["default_values"].keys():
      defaults[param] = cfg["default_values"][param]
    else:
      print(f"WARNING: Default value for {param} not set")
  for param in cfg["nuisances"]:
    defaults[param] = 0.0
  for rate in cfg["inference"]["rate_parameters"]:
    defaults[f"mu_{rate}"] = 1.0
  return defaults


def GetDefaultsInModel(file_name, cfg, include_rate=False, include_lnN=False):
  defaults = GetDefaults(cfg)
  if file_name == "combined":
    return defaults
  params_in_model = GetParametersInModel(file_name, cfg, include_rate=include_rate, include_lnN=include_lnN)
  return {k:v for k, v in defaults.items() if k in params_in_model}


def GetFilesInModel(file_name, cfg):
  parameters_in_model = []
  for v in cfg["models"][file_name]["density_models"]:
    parameters_in_model = list(set(parameters_in_model + [v["file"]]))
  for v in cfg["models"][file_name]["regression_models"]:
    parameters_in_model = list(set(parameters_in_model + [v["file"]]))
  return parameters_in_model


def GetParametersInModel(file_name, cfg, only_density=False, only_regression=False, include_rate=False, include_lnN=False):
  parameters_in_model = []
  if not only_regression:
    for v in cfg["models"][file_name]["density_models"]:
      parameters_in_model = list(set(parameters_in_model + v["parameters"]))
  if not only_density:
    for v in cfg["models"][file_name]["regression_models"]:
      parameters_in_model = list(set(parameters_in_model + [v["parameter"]]))
  if include_rate and file_name in cfg["inference"]["rate_parameters"]:
    parameters_in_model.append(f"mu_{file_name}")
  if include_lnN:
    parameters_in_model += list(cfg["inference"]["lnN"].keys())
  return parameters_in_model


def GetValidationLoop(cfg, file_name, include_rate=False, include_lnN=False, only_density=False, only_regression=False):
  if file_name == "combined":
    return cfg["validation"]["loop"]
  parameters_in_model = GetParametersInModel(file_name, cfg, include_rate=include_rate, include_lnN=include_lnN, only_density=only_density, only_regression=only_regression)
  if len(parameters_in_model) == 0:
    return [{}]
  loop_with_parameters = []
  for value in cfg["validation"]["loop"]:
    loop_with_parameters.append({k:v for k, v in value.items() if k in parameters_in_model})
  loop_with_unique_parameters = []
  for v in loop_with_parameters:
    if v not in loop_with_unique_parameters:
      loop_with_unique_parameters.append(v)

  return loop_with_unique_parameters


def GetCombinedValdidationIndices(cfg, file_name, val_ind):
  if file_name != "combined":
    return {file_name : val_ind}
  combined_val = GetValidationLoop(cfg, file_name)[val_ind]
  indices = {}
  for fn in GetModelFileLoop(cfg):
    indiv_val_loop = GetValidationLoop(cfg, fn)
    for ind, val in enumerate(indiv_val_loop):
      skimmed_combined_val = {k:v for k,v in combined_val.items() if k in val.keys()}
      if val == skimmed_combined_val:
        indices[fn] = ind
        break
  return indices


def InitiateDensityModel(architecture, file_loc, options={}, test_name=None):

  if architecture["type"] == "BayesFlow":

    from bayes_flow_network import BayesFlowNetwork
    network = BayesFlowNetwork(
      f"{file_loc}/X_train.parquet",
      f"{file_loc}/Y_train.parquet", 
      f"{file_loc}/wt_train.parquet",
      f"{file_loc}/X_{test_name}.parquet" if test_name is not None else None,
      f"{file_loc}/Y_{test_name}.parquet" if test_name is not None else None,
      f"{file_loc}/wt_{test_name}.parquet" if test_name is not None else None,
      options = {
        **{k:v for k,v in architecture.items() if k!="type"},
        **options
      }
    )

  elif architecture["type"] == "Benchmark":

    import importlib
    module = importlib.import_module(architecture["benchmark"])
    module_class = getattr(module, architecture["benchmark"])
    network = module_class()
    network.file_name = options["file_name"]

  return network


def InitiateRegressionModel(architecture, file_loc, options={}, test_name=None):

  if architecture["type"] == "FCNN":

    from fcnn_network import FCNNNetwork
    network = FCNNNetwork(
      f"{file_loc}/X_train.parquet",
      f"{file_loc}/y_train.parquet", 
      f"{file_loc}/wt_train.parquet",
      f"{file_loc}/X_{test_name}.parquet" if test_name is not None else None,
      f"{file_loc}/y_{test_name}.parquet" if test_name is not None else None,
      f"{file_loc}/wt_{test_name}.parquet" if test_name is not None else None,
      options = {
        **{k:v for k,v in architecture.items() if k!="type"},
        **options
      }
    )  

  return network


def SkipNonData(cfg, file_name, data_type, val_ind):

  if data_type != "data":
    return False

  if (file_name == "combined" or len(list(cfg["models"].keys())) == 1) and val_ind == 0:
    return False

  return True

  
def SkipNonDensity(cfg, file_name, val_info, skip_non_density=True):
  if not skip_non_density:
    return False
  if file_name == "combined":
    return True  

  params_in_density = GetParametersInModel(file_name, cfg, only_density=True)
  defaults_not_in_model = GetDefaultsInModel(file_name, cfg)

  default = True
  for k,v in val_info.items():
    if k in params_in_density: continue
    if v != defaults_not_in_model[k]:
      default = False
      break

  return not default


def GetFreezeLoop(freeze, val_info, file_name, cfg, column=None, include_rate=False, include_lnN=False, loop_over_nuisances=False, loop_over_rates=False, loop_over_lnN=False):

  val_info_with_defaults = GetDefaultsInModel(file_name, cfg, include_rate=include_rate, include_lnN=include_lnN)

  freeze_loop = []
  if not (freeze == "all-but-one"):

    freeze_loop += [{
      "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in freeze.split(",")} if freeze is not None else {},
      "extra_name" : "",
    }]

  elif len(val_info_with_defaults.keys()) < 2:
    freeze_loop += [{
      "freeze" : {},
      "extra_name" : "",
    }]  

  else:

    for k, v in val_info.items():
      val_info_with_defaults[k] = v
    for col in GetParameterLoop(file_name, cfg, include_nuisances=loop_over_nuisances, include_rate=loop_over_rates, include_lnN=loop_over_lnN):
      if column is not None:
        if col != column:
          continue
      freeze_loop += [{
        "freeze" : {k: v for k, v in val_info_with_defaults.items() if k != col},
        "extra_name" : f"_floating_only_{col}",
      }]

  return freeze_loop

    
def GetParameterLoop(file_name, cfg, include_nuisances=False, include_rate=False, include_lnN=False, include_per_model_rate=False, include_per_model_lnN=False):

  if not (file_name == "combined" or len(list(cfg["models"].keys())) == 1):
    if include_rate and not include_per_model_rate:
      include_rate = False
    if include_lnN and not include_per_model_lnN:
      include_lnN = False

  defaults_in_model = GetDefaultsInModel(file_name, cfg, include_rate=include_rate, include_lnN=include_lnN)  
  par_loop = [i for i in cfg["pois"] if i in defaults_in_model.keys()]
  par_loop += [f"mu_{i}" for i in cfg["inference"]["rate_parameters"] if f"mu_{i}" in defaults_in_model.keys()]
  if include_nuisances:
    par_loop += [i for i in cfg["nuisances"] if i in defaults_in_model.keys()]

  return par_loop


def GetScanArchitectures(cfg, data_output="data/", write=True):
  """
  Generate scan architectures based on configurations provided in a YAML file.

  Parameters
  ----------
  cfg : str
      Path to the YAML configuration file containing scan architecture parameters.
  data_output : str, optional
      Directory where generated YAML files will be saved. Defaults to "data/".
  write : bool, optional
      Whether to write the generated YAML files to disk. Defaults to True.

  Returns
  -------
  list
      List of paths to the generated YAML files.
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


def GetSnakeMakeStepLoop(input_cfg, output_cfg=[]):

  for step in input_cfg:

    if "step" in step.keys():
      output_cfg.append(step)

    elif "workflow" in step.keys():
      with open(step["workflow"], 'r') as yaml_file:
        workflow = yaml.load(yaml_file, Loader=yaml.FullLoader)
      output_cfg = GetSnakeMakeStepLoop(workflow, output_cfg=output_cfg)

  return output_cfg


def GetYName(ur, purpose="plot", round_to=2, prefix=""):
  """
  Generate a formatted label for a given unique row.

  Parameters
  ----------
  ur : list or pd.DataFrame
      List or DataFrame representing the unique row.
  purpose : str, optional
      Purpose of the label. Can be "plot" (default) or "file".
  round_to : int, optional
      Number of decimal places to round the values. Defaults to 2.
  prefix : str, optional
      Prefix to prepend to the generated label. Defaults to "".

  Returns
  -------
  str or None
      Formatted label for the given unique row, or None if the row is empty.
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


def LoadConfig(config_name):

  with open(config_name, 'r') as yaml_file:
    cfg_before = yaml.load(yaml_file, Loader=yaml.FullLoader)

  if "export_string" in cfg_before.keys():
    for k,v in cfg_before["export_string"].items():
      os.environ[k] = v
  with open(config_name, 'r') as yaml_file:
    content = os.path.expandvars(yaml_file.read())  # Replace ${VAR} with environment value
    cfg = yaml.safe_load(content)

  # if missing set some defaults
  if "inference" not in cfg.keys():
    cfg["inference"] = {}
  if "rate_parameters" not in cfg["inference"].keys():
    cfg["inference"]["rate_parameters"] = []
  if "lnN" not in cfg["inference"].keys():
    cfg["inference"]["lnN"] = {}
  if "nuisance_constraints" not in cfg.keys():
    cfg["nuisance_constraints"] = []
  for k, v in cfg["models"].items():
    if "regression_models" not in v.keys():
      cfg["models"][k]["regression_models"] = []
    if "density_models" not in v.keys():
      cfg["models"][k]["density_models"] = []
    for val_ind, val in enumerate(cfg["models"][k]["regression_models"]):
      if "n_copies" not in val.keys():
        cfg["models"][k]["regression_models"][val_ind]["n_copies"] = 1
    for val_ind, val in enumerate(cfg["models"][k]["density_models"]):
      if "n_copies" not in val.keys():
        cfg["models"][k]["density_models"][val_ind]["n_copies"] = 1
  for k, v in cfg["files"].items():
    if "pre_calculate" not in v.keys():
      cfg["files"][k]["pre_calculate"] = {}
    if "post_calculate_selection" not in v.keys():
      cfg["files"][k]["post_calculate_selection"] = None
    if "weight_shifts" not in v.keys():
      cfg["files"][k]["weight_shifts"] = {}

  return cfg


def MakeDictionaryEntry(dictionary, keys, val):
  """
  Make a dictionary entry.

  Parameters
  ----------
  dictionary : dict
      Dictionary.
  keys : list
      List of keys.
  val : object
      Value.

  Returns
  -------
  dict
      Dictionary containing the entry.
  """

  if len(keys) > 1:
    if keys[0] not in dictionary.keys():
      dictionary[keys[0]] = {}
    dictionary[keys[0]] = MakeDictionaryEntry(dictionary[keys[0]], keys[1:], val)
  else:
    dictionary[keys[0]] = val
  return dictionary


def MakeDirectories(file_loc):
  """
  Make directories.

  Parameters
  ----------
  file_loc : str
      File location.

  Returns
  -------
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


def Resample(datasets, weights, n_samples=None, seed=42):
  """
  Resamples datasets based on provided weights.

  Parameters
  ----------
  datasets : list or np.ndarray
      List of datasets or a single numpy array to be resampled.
  weights : np.ndarray
      Array of weights corresponding to each sample in the datasets.
  n_samples : int, optional
      Number of samples to generate after resampling. If None, defaults to the length of weights.
  seed : int, optional
      Seed for the random number generator. Defaults to 42.

  Returns
  -------
  tuple
      Tuple containing:
          - resampled_datasets (list or np.ndarray): Resampled datasets based on the weights.
          - resampled_weights (np.ndarray): Weights corresponding to the resampled datasets.
  """

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
  if do_positive and do_negative: 
    positive_fraction = np.sum(weights[positive_indices])/np.sum(weights)
    n_positive_samples = int(round(positive_fraction*n_samples,0))
    n_negative_samples = n_positive_samples - n_samples
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


def RoundToSF(num, sig_figs):
  """
  Round a number to a specified number of significant figures.

  Parameters
  ----------
  num : float
      The number to be rounded.
  sig_figs : int
      Number of significant figures to round to.

  Returns
  -------
  float
      The rounded number to the specified number of significant figures.
  """
  if num == 0:
    return 0
  else:
    rounded_number = round(num, sig_figs)
    decimal_places = len(str(rounded_number).rstrip('0').split(".")[1])
    return round(num, decimal_places)


def SetupSnakeMakeFile(args, default_args, main):
  """
  Setup a SnakeMake file based on configuration and input arguments.

  Parameters
  ----------
  args : object
      Input arguments object containing configuration details.
  default_args : object
      Default arguments object for fallback values.
  main : function
      Main function to execute for each step defined in SnakeMake configuration.

  Returns
  -------
  str
      File path of the generated SnakeMake file.
  """

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

  snakemake_loop = GetSnakeMakeStepLoop(snakemake_cfg)

  # Make snakemake file
  args.make_snakemake_inputs = True
  for step_info in snakemake_loop:
    args_copy = copy.deepcopy(args)
    args_copy.step = step_info["step"]
    if "submit" in step_info.keys():
      args_copy.submit = step_info["submit"]
    if "run_options" in step_info.keys():
      for arg_name, arg_val in step_info["run_options"].items():
        setattr(args_copy, arg_name, arg_val)

    # Open config when available
    if not args_copy.step in ["MakeBenchmark"] and clear_file:
      with open(args_copy.cfg, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
      snakemake_file = f"jobs/{cfg['name']}/innfer_SnakeMake.txt"
      if os.path.isfile(snakemake_file):
        os.system(f"rm {snakemake_file}")
      clear_file = False

    # Write info to file
    main(args_copy, default_args)

  # make final outputs
  with open(args_copy.cfg, 'r') as yaml_file:
    cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
  snakemake_file = f"jobs/{cfg['name']}/innfer_SnakeMake.txt"

  # Find outputs
  with open(snakemake_file, 'r') as file:
    for line in file:
      line = line.rstrip()
      all_lines.append(line)
      if line.startswith("  shell:") or line.startswith("  input:") or line.startswith("  params:") or line.startswith("  threads:") or line == "":
        output_line = False
        continue
      elif line.startswith("  output:"):
        output_line = True
        continue
      if output_line:
        if "," == line[-1]:
          rules_all.append(line)
        else:
          rules_all.append(line+",")
  rules_all[-1] = rules_all[-1][:-1]

  # Check if inputs and outputs are the same, if so then add dummy
  output_line = False
  input_line = False
  sweep = False
  for line_ind, line in enumerate(all_lines):
    if line.startswith("  input:"):
      input_line = True
      sweep = False
      inputs = []
    elif line.startswith("  output:"):
      output_line = True
      input_line = False
      sweep = False
      outputs = []
      output_ind = line_ind
    elif line.startswith("  shell:"):
      output_line = False
      input_line = False
      sweep = True
    elif input_line:
      inputs.append(line.replace(" ","").replace('"','').replace("'","").replace(",",""))
    elif output_line:
      outputs.append(line.replace(" ","").replace('"','').replace("'","").replace(",",""))
    elif sweep:
      for output in outputs:
        if output in inputs:
          all_lines[line_ind] = all_lines[line_ind][:-1] + f"; echo -n '' > {output.replace('.','_dummy.')}" + all_lines[line_ind][-1]
          for change_ind in range(output_ind, len(all_lines)):
            all_lines[change_ind] = all_lines[change_ind].replace(output, output.replace(".","_dummy."))
      sweep = False
    
  # Make snakemake file
  rules = rules_all+[""]+all_lines
  from batch import Batch
  b_rules = Batch()
  b_rules._CreateJob(rules, snakemake_file)   

  return snakemake_file


def StringToFile(string):
  """
  Convert a string into a format suitable for file naming.

  Parameters
  ----------
  string : str
      Input string to be converted.

  Returns
  -------
  str
      Converted string formatted for file naming.
  """

  string = string.replace(",","_").replace(";","_").replace(">=","_geq_").replace("<=","_leq_").replace(">","_g_").replace("<","_l_").replace("!=","_noteq_").replace("=","_eq_")
  if string.count(".") > 1:
    string = f'{"p".join(string.split(".")[:-1])}.{string.split(".")[-1]}'
  return string


def Translate(key, translation_file="configs/translate/translate.yaml"):
  val = GetDictionaryEntryFromYaml(translation_file, [key])
  if val is not None:
    return val
  else:
    return key