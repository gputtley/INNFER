import yaml

import numpy as np

from useful_functions import MakeDirectories, GetYName

class SummaryChiSquared():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.val_loop = None

    self.file_name = "scan_results"
    self.data_input = "data"
    self.data_output = "data"
    self.column_loop = []
    self.verbose = True

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

    # Open results
    if self.verbose:
      print("- Loading in results")
    results = {}
    for ind, info in enumerate(self.val_loop):
      info_name = GetYName(info["row"],purpose="file")
      for col in self.column_loop:
        if col in self.freeze.keys():
          continue
        with open(f"{self.data_input}/{self.file_name}_{col}_{ind}.yaml", 'r') as yaml_file:
          scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
        #crossings = {k:v/scan_results_info["row"][scan_results_info["columns"].index(col)] for k,v in scan_results_info["crossings"].items()}
        chi_squared = ((scan_results_info["crossings"][0]-scan_results_info["row"][scan_results_info["columns"].index(col)])**2) # numerator
        if scan_results_info["crossings"][0] < 1.0:
          chi_squared /= (scan_results_info["crossings"][1]-scan_results_info["crossings"][0])**2
        else:
          chi_squared /= (scan_results_info["crossings"][0]-scan_results_info["crossings"][-1])**2

        if col not in results.keys():
          results[col] = {info_name:chi_squared}
        else:
          results[col][info_name] = chi_squared

    # Make combined chi squared / ndof values 
    total_sum = 0.0
    total_count = 0.0
    for Y_col, poi_dict in results.items():
      total_sum += float(np.sum(list(poi_dict.values())))
      total_count += len(list(poi_dict.values()))
      results[Y_col]["all"] = float(np.sum(list(poi_dict.values())))/len(list(poi_dict.values()))
    results["all"] = total_sum/total_count

    # Write to yaml
    file_name = f"{self.data_output}/summary_chi_squared.yaml"
    MakeDirectories(file_name)
    with open(file_name, 'w') as yaml_file:
      yaml.dump(results, yaml_file, default_flow_style=False)
    print(f"Created {file_name}")


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [
      f"{self.data_output}/summary_chi_squared.pdf"
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for ind, info in enumerate(self.val_loop):
      info_name = GetYName(info["row"],purpose="plot",prefix="y=")
      for col in self.column_loop:
        if col in self.freeze.keys():
          continue
        inputs.append(f"{self.data_input}/{self.file_name}_{col}_{ind}.yaml")

    return inputs
