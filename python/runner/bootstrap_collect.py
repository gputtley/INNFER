import yaml
import numpy as np

from useful_functions import MakeDirectories

class BootstrapCollect():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.number_of_bootstraps = None
    self.columns = None

    self.data_input = "data"
    self.data_output = "data"    
    self.extra_file_name = ""

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.extra_file_name != "":
      self.extra_file_name = f"_{self.extra_file_name}"

    self.bootstrap_files = [f"{self.data_input}/best_fit{self.extra_file_name}_{i}.yaml" for i in range(self.number_of_bootstraps)]

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    # Load scan results
    bootstrap_results = {}
    for bootstrap_ind, bootstrap_file in enumerate(self.bootstrap_files):      

      with open(bootstrap_file, 'r') as yaml_file:
        bootstrap_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      for column in self.columns:

        if column not in bootstrap_results.keys():
          bootstrap_results[column] = {
            "columns" : self.columns,
            "varied_column" : column,
            "row" : bootstrap_results_info["row"],
            "results" : []
            }
        
        bootstrap_results[column]["results"].append(bootstrap_results_info["best_fit"][bootstrap_results_info["columns"].index(column)])
        
    for column in self.columns:

      # Make mean, std and crossings
      bootstrap_results[column]["mean"] = float(np.mean(bootstrap_results[column]["results"]))
      bootstrap_results[column]["std"] = float(np.std(bootstrap_results[column]["results"]))
      bootstrap_results[column]["crossings"] = {
        -2 : bootstrap_results[column]["mean"] - (2 * bootstrap_results[column]["std"]),
        -1 : bootstrap_results[column]["mean"] - (1 * bootstrap_results[column]["std"]),
        0 : bootstrap_results[column]["mean"],
        1 : bootstrap_results[column]["mean"] + (1 * bootstrap_results[column]["std"]),
        2 : bootstrap_results[column]["mean"] + (2 * bootstrap_results[column]["std"])
      }

      # Dump to yaml
      file_name = f"{self.data_output}/bootstrap_results_{column}{self.extra_file_name}.yaml"
      MakeDirectories(file_name)
      print(f"Created {file_name}")
      with open(file_name, 'w') as yaml_file:
        yaml.dump(bootstrap_results[column], yaml_file, default_flow_style=False)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    for column in self.columns:
      outputs.append(f"{self.data_output}/bootstrap_results_{column}{self.extra_file_name}.yaml")
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = self.bootstrap_files
    return inputs

  def _FindCrossings(self, x, y, crossings=[1, 2]):
    """
    Finds the intersection points for the likelihood curve.

    Args:
        x (list): List of X values.
        y (list): List of likelihood values.
        best_fit (float): The best-fit value for the parameter.
        crossings (list): List of crossing points to find (default is [1, 2]).

    Returns:
        dict: Dictionary containing crossing points and their corresponding values.

    """
    values = {}
    values[0] = float(x[y.index(min(y))])
    for crossing in crossings:
      for sign in [-1, 1]:
        condition = np.array(x) * sign > values[0] * sign
        filtered_x = np.array(x)[condition]
        filtered_y = np.array(y)[condition]
        sorted_indices = np.argsort(filtered_y)
        filtered_x = filtered_x[sorted_indices]
        filtered_y = filtered_y[sorted_indices]
        filtered_x = filtered_x.astype(np.float64)
        filtered_y = filtered_y.astype(np.float64)
        if len(filtered_y) > 1:
          if crossing**2 > min(filtered_y) and crossing**2 < max(filtered_y):
            values[sign * crossing] = float(np.interp(crossing**2, filtered_y, filtered_x))
    return values