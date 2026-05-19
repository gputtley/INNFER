import copy
import os
import yaml

from useful_functions import (
    GetCategoryLoop, 
    GetDefaultsInModel,
    GetModelFileLoop, 
    LoadConfig,
    MakeDirectories
)

data_dir = str(os.getenv("PREP_DATA_DIR"))

class set_binned_to_nominal():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.cfg = None
    self.options = {}

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    self.parameters = self.options["parameters"].split(",") if "parameters" in self.options else []


  def Run(self):
    """
    Run the code utilising the worker classes
    """
    # Load the config
    cfg = LoadConfig(self.cfg)

    # Set up extra name for copying files if using copies
    copy_name = "_before_binned_to_nominal"

    for file_name in GetModelFileLoop(cfg):
      for category in GetCategoryLoop(cfg):

        # Get default index of the binned fit
        defaults = GetDefaultsInModel(file_name, cfg, category=category)
        poi = cfg["pois"][0]
        bins = cfg["inference"]["binned_fit"]["shape_poi_values"]
        if poi not in defaults or defaults[poi] not in bins:
          continue
        default_ind = bins.index(defaults[poi])

        # Loop over parameters
        for parameter in self.parameters:

          # Open the default file
          default_file_name = f"{data_dir}/{cfg['name']}/PreProcess/{file_name}/{category}/parameters_binned_fit_inputs_{parameter}_{default_ind}.yaml"
          with open(default_file_name, 'r') as f:
            default_data = yaml.safe_load(f)

          for bin_ind, bin_val in enumerate(bins):
            if bin_ind == default_ind: continue
            no_default_file_name = f"{data_dir}/{cfg['name']}/PreProcess/{file_name}/{category}/parameters_binned_fit_inputs_{parameter}_{bin_ind}.yaml"

            # make copy of file_name
            copy_file_name = no_default_file_name.replace('.yaml', f'{copy_name}.yaml')
            if not os.path.exists(copy_file_name):
              os.system(f"cp {no_default_file_name} {copy_file_name}")

            updated_default_data = copy.deepcopy(default_data)
            for file_bin_ind, bin in enumerate(default_data["binned_fit_input"]):
              for yield_key, yield_val in bin["yields"].items():
                updated_default_data["binned_fit_input"][file_bin_ind]["yields"][bin_val] = yield_val
                del updated_default_data["binned_fit_input"][file_bin_ind]["yields"][yield_key]

            # Save the updated file
            print(f"- Updating {no_default_file_name}")
            with open(no_default_file_name, 'w') as f:
              yaml.dump(updated_default_data, f)

    # Write a dummy file to indicate that the code has run
    for parameter in self.parameters:
      dummy_name = f"{data_dir}/{cfg['name']}/PreProcess/set_binned_to_nominal_done_{parameter}.txt"
      MakeDirectories(dummy_name)
      with open(dummy_name, 'w') as f:
        f.write("Done")

  def Outputs(self):
    """
    Return a list of outputs given by class
    """

    cfg = LoadConfig(self.cfg)

    outputs = []
    for parameter in self.parameters:
      outputs.append(f"{data_dir}/{cfg['name']}/PreProcess/set_binned_to_nominal_done_{parameter}.txt")

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    # Load the config
    cfg = LoadConfig(self.cfg)

    # Set up extra name for copying files if using copies
    for file_name in GetModelFileLoop(cfg):
      for category in GetCategoryLoop(cfg):
        for parameter in self.parameters:

          defaults = GetDefaultsInModel(file_name, cfg, category=category)
          poi = cfg["pois"][0]
          bins = cfg["inference"]["binned_fit"]["shape_poi_values"]
          if poi not in defaults or defaults[poi] not in bins:
            continue

          for bin_ind in range(len(bins)):
            no_default_file_name = f"{data_dir}/{cfg['name']}/PreProcess/{file_name}/{category}/parameters_binned_fit_inputs_{parameter}_{bin_ind}.yaml"
            inputs.append(no_default_file_name)
        
    return inputs

        