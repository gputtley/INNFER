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

  if args.data_type == "sim":
    data_input = {k:[f"data/{cfg['name']}/PreProcess/{k}/val_ind_{v}/{i}_{args.sim_type}.parquet" for i in ["X","wt"]] for k,v in GetCombinedValdidationIndices(cfg, file_name, val_ind).items()}
  elif args.data_type == "asimov":
    data_input = {k:[f"data/{cfg['name']}/MakeAsimov{args.extra_infer_dir_name}/{k}/val_ind_{v}/asimov.parquet"] for k,v in GetCombinedValdidationIndices(cfg, file_name, val_ind).items()}


  if args.likelihood_type == "binned":
    print("Not implemented yet")

  common_config = {
    "density_models" : {k:GetModelLoop(cfg, model_file_name=k, only_density=True)[0] for k in ([file_name] if file_name != "combined" else GetModelFileLoop(cfg))},
    "regression_models" : {k:GetModelLoop(cfg, model_file_name=k, only_regression=True) for k in ([file_name] if file_name != "combined" else GetModelFileLoop(cfg))},
    "model_input" : f"models/{cfg['name']}",
    "parameters" : {k:f"data/{cfg['name']}/PreProcess/{k}/parameters.yaml" for k in GetCombinedValdidationIndices(cfg, file_name, val_ind).keys()},
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
    "Y_columns" : list(defaults_in_model),
    "Y_columns_per_model" : {k: GetParametersInModel(k, cfg) for k in ([file_name] if file_name != "combined" else GetModelFileLoop(cfg))},
    "only_density" : args.only_density,
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

        # Generate random numbers
        #sampled = np.random.RandomState().rand(n_samples)
        #sampled = np.linspace(0,1,num=n_samples) + ((np.random.RandomState().rand(n_samples)-0.5)/(n_samples))
        #np.random.shuffle(sampled)
        #print(sampled)
        #print(np.interp(sampled, cdf_vals, param_values))

        # Inverse transform sampling
        df.loc[indices,col] = np.interp(sampled, cdf_vals, param_values)

        #print("----------------")
        #print(np.histogram(sampled, bins=20))
        #print(np.histogram(sampled, weights=df.loc[indices,wt_name], bins=20))
        #print(np.histogram(df.loc[indices,col], bins=20))
        #print(np.histogram(df.loc[indices,col], weights=df.loc[indices,wt_name], bins=20))
        #print(np.histogram(df.loc[:,col], weights=df.loc[:,wt_name], bins=20))

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

def GetCombinedValidateLoop(cfg, parameters):
  """
  Generate a list of dictionaries representing parameter combinations for combined validation loops.

  Parameters
  ----------
  cfg : dict
      Configuration dictionary containing "pois", "nuisances", "inference", and "validation".
  parameters : dict
      Dictionary containing parameter information.

  Returns
  -------
  list
      List of dictionaries representing parameter combinations for combined validation loops.
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
    for file in GetFileLoop(cfg):
      if poi not in parameters[file]["unique_Y_values"]: continue
      for val in parameters[file]["unique_Y_values"][poi]:
        if val not in unique_values_for_pois[-1]:
          unique_values_for_pois[-1].append(val)

    average_poi = sum(unique_values_for_pois[-1])/len(unique_values_for_pois[-1])
    initial_poi_guess.append(min(unique_values_for_pois[-1], key=lambda x: abs(x - average_poi)))

  for nuisance in cfg["nuisances"]:
    nuisances.append(nuisance)
    #unique_values_for_nuisances.append([0.0])
    unique_values_for_nuisances.append([])
    for file in GetFileLoop(cfg):
      if nuisance not in parameters[file]["Y_columns"]: continue
      for nuisance in parameters[file]["unique_Y_values"][nuisance]:
        if val not in unique_values_for_nuisances[-1]:
          unique_values_for_nuisances[-1].append(nuisance)

  initial_best_fit_guess = np.array(initial_poi_guess+[0]*(len(nuisances)))

  for poi_values in list(product(*unique_values_for_pois)): 
    if not cfg.get("validation", {}).get("off_diagonal_nuisances", False):
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
    else:
      for nuisance_values in list(product(*unique_values_for_nuisances)):
        row = np.array(list(poi_values)+list(nuisance_values))
        columns = pois+nuisances
        sorted_columns = sorted(columns)
        sorted_row = [row[columns.index(col)] for col in sorted_columns]
        sorted_initial_best_fit_guess = [initial_best_fit_guess[columns.index(col)] for col in sorted_columns]
        val_loop.append({
          "row" : pd.DataFrame([sorted_row], columns=sorted_columns, dtype=np.float64),
          "initial_best_fit_guess" : pd.DataFrame([sorted_initial_best_fit_guess], columns=sorted_columns, dtype=np.float64),
        })

  return val_loop

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

def GetFileLoop(cfg):
  return list(cfg["files"].keys())

def GetModelFileLoop(cfg, with_combined=False):
  model_files = list(cfg["models"].keys())
  if with_combined: 
    model_files += ["combined"]
  return model_files


def GetModelLoop(cfg, only_density=False, only_regression=False, model_file_name=None):
  models = []
  for k, v in cfg["models"].items():

    if model_file_name is not None:
      if model_file_name != k:
        continue

    if not only_regression:
      models.append(
        {
          "type" : "density",
          "file_loc" : f"data/{cfg['name']}/PreProcess/{k}/density/",
          "name" : f"density_{k}",
          "parameters" : f"data/{cfg['name']}/PreProcess/{k}/parameters.yaml",
          "parameter" : None,
        }
      )
    for value in v["regression_models"]:
      if not only_density:
        models.append(
          {
            "type" : "regression",
            "file_loc" : f"data/{cfg['name']}/PreProcess/{k}/regression/{value['parameter']}",
            "name" : f"regression_{k}_{value['parameter']}",
            "parameters" : f"data/{cfg['name']}/PreProcess/{k}/parameters.yaml",
            "parameter" : value['parameter'],
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

def GetValidationLoop(cfg, file_name, include_rate=False, include_lnN=False):
  if file_name == "combined":
    return cfg["validation"]["loop"]
  parameters_in_model = GetParametersInModel(file_name, cfg, include_rate=include_rate, include_lnN=include_lnN)
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

    #val_info_with_defaults = GetDefaultsInModel(file_name, cfg, include_rate=include_rate, include_lnN=include_lnN)
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

#def GetFreezeLoop(freeze, val_info, column=None):
#  freeze_loop = []
#  if not (freeze == "all-but-one"):
#    freeze_loop += [{
#      "freeze" : {k.split("=")[0] : float(k.split("=")[1]) for k in freeze.split(",")} if freeze is not None else {},
#      "extra_name" : "",
#    }]
#  elif len(val_info['row'].columns) < 2:
#    freeze_loop += [{
#      "freeze" : {},
#      "extra_name" : "",
#    }]  
#  else:
#    for col in val_info['row'].columns:
#      if column is not None:
#        if col != column:
#          continue
#      freeze_loop += [{
#        "freeze" : {c:float(val_info['row'].loc[:,c].iloc[0]) for c in val_info['row'].columns if c != col},
#        "extra_name" : f"_floating_only_{col}",
#      }]
#  return freeze_loop

#def GetFileLoop(cfg):
#  
#  split_nuisances = False
#  if "split_nuisance_models" in cfg.keys():
#    split_nuisances = cfg["split_nuisance_models"]  
#
#  if not split_nuisances:
#    return list(cfg["files"].keys())
#  else:
#    split_nuisance_list = []
#    for key in cfg["files"].keys():
#      split_nuisance_list.append(key)
#      for nuisance in cfg["nuisances"]:
#        split_nuisance_list.append(f"{key}_{nuisance}")
#    return split_nuisance_list
    
def GetNuisanceLoop(cfg, parameters):
  """
  Generate a list of dictionaries representing parameter combinations for nuisance loops.

  Parameters
  ----------
  cfg : dict
      Configuration dictionary containing "pois" and "nuisances".
  parameters : dict
      Dictionary containing parameter information.

  Returns
  -------
  list
      List of dictionaries representing parameter combinations for nuisance loops.
  """

  nuisance_loop = []

  for nuisance in cfg["nuisances"]:
    if nuisance not in parameters["Y_columns"]: continue
    nuisance_freeze={k:0.0 for k in cfg["nuisances"] if k != nuisance and k in parameters["Y_columns"]}
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

"""
def GetParameterLoop(val_info, cfg, args):

  par_loop = []
  for i in val_info["initial_best_fit_guess"].columns:
    cor_par = cfg["pois"]
    if args.loop_over_rates:
      cor_par += ["mu_"+rp for rp in cfg["inference"]["rate_parameters"]]
    if args.loop_over_nuisances:
      cor_par += cfg["nuisances"]
  return [i for i in val_info["initial_best_fit_guess"].columns if i in cor_par]
"""

def GetParameterLoop(file_name, cfg, include_nuisances=False, include_rate=False, include_lnN=False):

  defaults_in_model = GetDefaultsInModel(file_name, cfg, include_rate=include_rate, include_lnN=include_lnN)  
  par_loop = [i for i in cfg["pois"] if i in defaults_in_model.keys()]
  par_loop += [f"mu_{i}" for i in cfg["inference"]["rate_parameters"] if f"mu_{i}" in defaults_in_model.keys()]
  if include_nuisances:
    par_loop += [i for i in cfg["nuisances"] if i in defaults_in_model.keys()]
  return par_loop



def GetPOILoop(cfg, parameters):
  """
  Generate a list of dictionaries representing parameter combinations for POI loops.

  Parameters
  ----------
  cfg : dict
      Configuration dictionary containing "pois" and "nuisances".
  parameters : dict
      Dictionary containing parameter information.

  Returns
  -------
  list
      List of dictionaries representing parameter combinations for POI loops.
  """

  poi_loop = []

  for poi in cfg["pois"]:
    nuisance_freeze = {k:0 for k in cfg["nuisances"] if k in parameters["Y_columns"]}
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

def GetValidateInfo(
    preprocess_loc, 
    models_loc, 
    cfg, 
    data_type = "sim", 
    skip_empty_Y = False
  ):
  """
  Generate validation information based on configuration and data type.

  Parameters
  ----------
  preprocess_loc : str
      Location of the preprocessed data.
  models_loc : str
      Location of the trained models.
  cfg : dict
      Configuration dictionary containing file names and parameters.
  data_type : str, optional
      Type of data ('sim', 'asimov', 'data'). Defaults to "sim".
  skip_empty_Y : bool, optional
      Whether to skip validation loops with empty target data. Defaults to False.

  Returns
  -------
  dict
      Dictionary containing validation information structured based on data_type.
  """

  parameters = {file_name: f"{preprocess_loc}/{file_name}/PreProcess/parameters.yaml" for file_name in GetFileLoop(cfg)}
  loaded_parameters = {file_name: yaml.load(open(file_loc), Loader=yaml.FullLoader) for file_name, file_loc in parameters.items()}
  if "val_loop" not in cfg["validation"].keys():
    val_loops = {file_name: GetValidateLoop(cfg, loaded_parameters[file_name]) for file_name in GetFileLoop(cfg)}
  else:
    val_loops = {file_name: [{"row": pd.DataFrame(val_loop["row"], columns=val_loop["columns"]),"initial_best_fit_guess": pd.DataFrame(val_loop["initial_best_fit_guess"], columns=val_loop["columns"])} for val_loop in cfg["validation"]["val_loop"][file_name]] for file_name in GetFileLoop(cfg)}

  models = {file_name: f"{models_loc}/{file_name}/{file_name}.h5" for file_name in GetFileLoop(cfg)}
  architectures = {file_name: f"{models_loc}/{file_name}/{file_name}_architecture.yaml" for file_name in GetFileLoop(cfg)}
  if len(GetFileLoop(cfg)) > 1:
    if "val_loop" not in cfg["validation"].keys():
      val_loops["combined"] = GetCombinedValidateLoop(cfg, loaded_parameters)
    else:
      val_loops["combined"] = [{"row": pd.DataFrame(val_loop["row"], columns=val_loop["columns"]),"initial_best_fit_guess": pd.DataFrame(val_loop["initial_best_fit_guess"], columns=val_loop["columns"])} for val_loop in cfg["validation"]["val_loop"]["combined"]]
    parameters["combined"] = copy.deepcopy(parameters)
    models["combined"] = copy.deepcopy(models)
    architectures["combined"] = copy.deepcopy(architectures)


  if skip_empty_Y and data_type != "data":
    val_loops = {k : v for k, v in val_loops.items() if len(list(v[0]["row"].columns)) > 0}

  if data_type in ["sim", "asimov"]:
    info = {
      "val_loops" : val_loops,
      "parameters" : parameters,
      "models" : models,
      "architectures" : architectures,
    }
  elif data_type in ["data"]:
    key = "combined" if len(list(val_loops.keys())) > 1 else list(val_loops.keys())[0]
    info = {
      "val_loops" : {key: [{"row" : None, "initial_best_fit_guess" : val_loops[key][0]["initial_best_fit_guess"]}]},
      "parameters" : {key : parameters[key]},
      "models" : {key : models[key]},
      "architectures" : {key : architectures[key]},
    }

  return info

def GetValidateLoop(cfg, parameters_file):
  """
  Generate a list of dictionaries representing parameter combinations for validation loops.

  Parameters
  ----------
  cfg : dict
      Configuration dictionary containing "pois" and "nuisances".
  parameters_file : dict
      Dictionary containing parameter information.

  Returns
  -------
  list
      List of dictionaries representing parameter combinations for validation loops.
  """

  val_loop = []

  pois = [poi for poi in cfg["pois"] if poi in parameters_file["Y_columns"]]
  unique_values_for_pois = [parameters_file["unique_Y_values"][poi] for poi in pois]
  nuisances = [nuisance for nuisance in cfg["nuisances"] if nuisance in parameters_file["Y_columns"]]
  unique_values_for_nuisances = [parameters_file["unique_Y_values"][nuisance] for nuisance in nuisances]
  average_pois = [sum(parameters_file["unique_Y_values"][poi])/len(parameters_file["unique_Y_values"][poi]) for poi in pois]
  initial_poi_guess = [min(unique_values_for_pois[ind], key=lambda x: abs(x - average_pois[ind])) for ind in range(len(pois))]
  initial_best_fit_guess = np.array(initial_poi_guess+[0.0]*(len(nuisances)))        

  for poi_values in list(product(*unique_values_for_pois)):
    if not cfg.get("validation", {}).get("off_diagonal_nuisances", False):
      nuisances_loop = [None] if len(nuisances) == 0 else copy.deepcopy(nuisances)
      for ind, nuisance in enumerate(nuisances_loop):
        nuisance_value_loop = [None] if len(nuisances) == 0 else unique_values_for_nuisances[ind]
        for nuisance_value in nuisance_value_loop:
          other_nuisances = [v for v in cfg["nuisances"] if v != nuisance and v in parameters_file["Y_columns"]]
          nuisance_in_row = [] if nuisance is None else [nuisance]
          nuisance_value_in_row = [] if nuisance_value is None else [nuisance_value]
          row = np.array(list(poi_values)+nuisance_value_in_row+[0]*len(other_nuisances))
          columns = pois+nuisance_in_row+other_nuisances
          sorted_columns = sorted(columns)
          sorted_row = [row[columns.index(col)] for col in sorted_columns]
          sorted_initial_best_fit_guess = [initial_best_fit_guess[columns.index(col)] for col in sorted_columns]
          val_loop.append({
            "row" : pd.DataFrame([sorted_row], columns=sorted_columns, dtype=np.float64),
            "initial_best_fit_guess" : pd.DataFrame([sorted_initial_best_fit_guess], columns=sorted_columns, dtype=np.float64),
          })
    else:
      for nuisance_values in list(product(*unique_values_for_nuisances)):
        row = np.array(list(poi_values)+list(nuisance_values))
        columns = pois+nuisances
        sorted_columns = sorted(columns)
        sorted_row = [row[columns.index(col)] for col in sorted_columns]
        sorted_initial_best_fit_guess = [initial_best_fit_guess[columns.index(col)] for col in sorted_columns]
        val_loop.append({
          "row" : pd.DataFrame([sorted_row], columns=sorted_columns, dtype=np.float64),
          "initial_best_fit_guess" : pd.DataFrame([sorted_initial_best_fit_guess], columns=sorted_columns, dtype=np.float64),
        })

  return val_loop

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

  # Make snakemake file
  args.make_snakemake_inputs = True
  for step_info in snakemake_cfg:
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

  # Make snakemake file
  rules = rules_all+[""]+all_lines
  from batch import Batch
  b_rules = Batch()
  b_rules._CreateJob(rules, snakemake_file)   

  return snakemake_file

def SplitValidationParameters(val_loops, file_name, val_ind, cfg):

  if file_name == "combined":
    output = {}
    combined_row = val_loops["combined"][val_ind]["row"]
    #for fn, val_loop in val_loops.items():
    for fn in cfg["files"].keys():
      if fn not in val_loops.keys():
        output[fn] = f"data/{cfg['name']}/{fn}/PreProcess/parameters.yaml"
      else:
        for indiv_val_ind, val_info in enumerate(val_loops[fn]):
          indiv_columns = [col for col in val_info["row"].columns if col in list(combined_row.columns)]
          indiv_row = val_info["row"].loc[:, indiv_columns]
          combined_row_with_indiv_columns = combined_row.loc[:, indiv_columns]
          if indiv_row.equals(combined_row_with_indiv_columns):
            if len(indiv_row.columns) > 0:
              output[fn] = f"data/{cfg['name']}/{fn}/SplitValidationFiles/val_ind_{indiv_val_ind}/parameters.yaml"
            else:
              output[fn] = f"data/{cfg['name']}/{fn}/PreProcess/parameters.yaml"
            break   

      #if fn == "combined": continue
      #for indiv_val_ind, val_info in enumerate(val_loop):
      #  indiv_columns = [col for col in val_info["row"].columns if col in list(combined_row.columns)]
      #  indiv_row = val_info["row"].loc[:, indiv_columns]
      #  combined_row_with_indiv_columns = combined_row.loc[:, indiv_columns]
      #  if indiv_row.equals(combined_row_with_indiv_columns):
      #    if len(indiv_row.columns) > 0:
      #      output[fn] = f"data/{cfg['name']}/{fn}/SplitValidationFiles/val_ind_{indiv_val_ind}/parameters.yaml"
      #    else:
      #      output[fn] = f"data/{cfg['name']}/{fn}/PreProcess/parameters.yaml"
      #    break
    return output
  else:
    return f"data/{cfg['name']}/{file_name}/SplitValidationFiles/val_ind_{val_ind}/parameters.yaml"

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