import yaml

from data_processor import DataProcessor
from useful_functions import MakeDirectories

class SimToDataFactors():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.parameters_input = {}
    self.data_input = {}
    self.groups = []
    self.processes = []
    self.data_output = None
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
    out = {"factor": {}, "sum": {}}

    for group in self.groups:

      sum_data = 0.0
      sum_total_sim = 0.0
      sum_process_sim = 0.0

      file_names = []

      # If all of group not in self.data_input.keys(), skip group
      if not all([cat in self.data_input.keys() for cat in group]):
        if self.verbose:
          print(f"Skipping group {group} as not all categories in data input")
        continue

      for cat in group:

        data_dp = DataProcessor([[x] for x in self.data_input[cat]], "parquet")
        data_cat = data_dp.GetFull(method="sum")
        sum_data += data_cat
        if "data" not in out["sum"].keys():
          out["sum"]["data"] = {}
        if cat not in out["sum"]["data"].keys():
          out["sum"]["data"][cat] = 0.0
        out["sum"]["data"][cat] += data_cat

        if "total_sim" not in out["sum"].keys():
          out["sum"]["total_sim"] = {}
        if cat not in out["sum"]["total_sim"].keys():
          out["sum"]["total_sim"][cat] = 0.0

        for file in self.parameters_input[cat]:
          with open(file, "r") as f:
            parameters = yaml.safe_load(f)

          file_name = parameters["file_name"]
          file_names.append(file_name)

          if file_name not in out["factor"].keys():
            out["factor"][file_name] = {}
          if cat not in out["factor"][file_name].keys():
            out["factor"][file_name][cat] = 1.0
          if file_name not in out["sum"].keys():
            out["sum"][file_name] = {}
          if cat not in out["sum"][file_name].keys():
            out["sum"][file_name][cat] = 0.0

        
          out["sum"][file_name][cat] += parameters["yields"]["nominal"]
          out["sum"]["total_sim"][cat] += parameters["yields"]["nominal"]
          sum_total_sim += parameters["yields"]["nominal"]
          if file_name in self.processes:
            sum_process_sim += parameters["yields"]["nominal"]

      # Add factors
      sum_other_sim = sum_total_sim - sum_process_sim
      sum_data_process = sum_data - sum_other_sim
      factor = sum_data_process/sum_process_sim
      for cat in group:
        for file_name in file_names:
          if file_name in self.processes:
            out["factor"][file_name][cat] = factor

    if self.verbose:
      for group_out in out:
        print(group_out)
        for file, cat_out in out[group_out].items():
          print(f"  {file}")
          for cat, factor in cat_out.items():
            print(f"    {cat}: {factor}")

    new_out = {}
    for group_out in out:
      for file, cat_out in out[group_out].items():
        for cat, factor in cat_out.items():
          if cat not in new_out.keys():
            new_out[cat] = {}
          if file not in new_out[cat].keys():
            new_out[cat][file] = {}
          new_out[cat][file][group_out] = factor


    for cat, cat_dict in new_out.items():

      # Save output
      fn = f"{self.data_output}_{cat}.yaml"
      MakeDirectories(fn)
      with open(fn, "w") as f:
        yaml.dump(cat_dict, f)
      if self.verbose:
        print(f"Created {fn}")



  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    for group in self.groups:
      if not all([cat in self.data_input.keys() for cat in group]): continue
      for cat in group:
        outputs += [f"{self.data_output}_{cat}.yaml"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    for _, files in self.data_input.items():
      inputs += files

    for _, files in self.parameters_input.items():
      inputs += files

    return inputs

        