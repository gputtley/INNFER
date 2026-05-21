import uproot
import yaml

import numpy as np
import pandas as pd

from useful_functions import GetDefaults, GetParametersInModel, LoadConfig

class ParametersToROOT():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.cfg = None
    self.open_cfg = None
    self.parameters = None
    self.data_output = "data/"
    self.verbose = False

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):
    """
    Run the code utilising the worker classes
    """
    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      cfg = LoadConfig(self.cfg)

    # Load the parameters
    parameters = {}
    for category, category_info in self.parameters.items():
      parameters[category] = {}
      for file_name, file_name_info in category_info.items():
        with open(file_name_info, 'r') as yaml_file:
          parameters[category][file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build yields
    poi = cfg["pois"][0]
    from infer import Infer
    infer_class = Infer()
    infer_class.Configure(
      {
        "parameters" : self.parameters,
        "binned_fit_morph_col" : poi,
        "inference_options" : cfg["inference"],
      }
    )

    hists = {}
    bins = {}
    for category, category_info in parameters.items():

      hists[category] = {}
      bins[category] = cfg["inference"]["binned_fit"]["input"][category]["binning"]
      yields = infer_class._BuildBinYields()[category]
      defaults = GetDefaults(cfg)
      base_dict = {k: [v] for k, v in defaults.items()}
      Y = pd.DataFrame(base_dict)

      for k, v in yields.items():

        parameters_in_model = GetParametersInModel(k, cfg, include_lnN=True, category=category)
        nuisances_in_model = [nuisance for nuisance in cfg["nuisances"] if nuisance in parameters_in_model]

        keys = list(parameters[category][k]["binned_fit_input"][0]["yields"].keys())
        if 'all' in keys:
          hists[category][k] = list(v(Y))

          for nuisance in nuisances_in_model:
            for nuisance_name, nuisance_value in {"Down": -1, "Up": 1}.items():
              key_dict = base_dict.copy()
              key_dict[nuisance] = [nuisance_value]
              key_Y = pd.DataFrame(key_dict)
              hists[category][f"{k}_{nuisance}{nuisance_name}"] = list(v(key_Y))

        else:
          for key in keys:
            key_dict = base_dict.copy()
            key_dict[poi] = [float(key)]
            key_Y = pd.DataFrame(key_dict)
            hists[category][f"{k}_{str(key).replace('.','')}"] = list(v(key_Y))

            for nuisance in nuisances_in_model:
              for nuisance_name, nuisance_value in {"Down": -1, "Up": 1}.items():
                key_nuisance_dict = key_dict.copy()
                key_nuisance_dict[nuisance] = [nuisance_value]
                key_Y = pd.DataFrame(key_nuisance_dict)
                hists[category][f"{k}_{str(key).replace('.','')}_{nuisance}{nuisance_name}"] = list(v(key_Y))

    # Write to ROOT file with directories as categories and histograms inside
    with uproot.recreate(self.data_output) as output_file:
      for category, category_hists in hists.items():
        bins_array = np.asarray(bins[category], dtype=np.float64)
        for hist_name, hist_values in category_hists.items():
          values_array = np.asarray(hist_values, dtype=np.float64)

          # Uproot writes TH1 histograms from (values, bin_edges)
          output_file[f"{category}/{hist_name}"] = values_array, bins_array

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [self.data_output]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [self.cfg]

    for _, category_info in self.parameters.items():
      for _, file_name_info in category_info.items():
        inputs.append(file_name_info)

    return inputs

        