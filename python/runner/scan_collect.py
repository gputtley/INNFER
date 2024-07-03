import yaml

import numpy as np

from useful_functions import MakeDirectories

class ScanCollect():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.number_of_scan_points = None
    self.column = None

    self.verbose = True
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

    self.scan_files = [f"{self.data_input}/scan_values_{self.column}{self.extra_file_name}_{i}.yaml" for i in range(self.number_of_scan_points)]

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    # Load scan results
    if self.verbose:
      print("- Collecting scan result files")
    scan_results = {"nlls":[],"scan_values":[]}
    for point_ind, scan_file in enumerate(self.scan_files):      

      with open(scan_file, 'r') as yaml_file:
        scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

      if point_ind == 0:
        scan_results["row"] = scan_results_info["row"]
        scan_results["columns"] = scan_results_info["columns"]
        scan_results["varied_column"] = scan_results_info["varied_column"]
      if None in scan_results_info["nlls"]: continue
      scan_results["scan_values"] += scan_results_info["scan_values"]
      scan_results["nlls"] += scan_results_info["nlls"]

    # Recheck minimum
    if self.verbose:
      print("- Rechecking the minimum")
    min_nll = min(scan_results["nlls"])
    scan_results["nlls"] = [i - min_nll for i in scan_results["nlls"]]

    # Get crossings
    if self.verbose:
      print("- Finding the intervals")
    scan_results["crossings"] = self._FindCrossings(scan_results["scan_values"], scan_results["nlls"], crossings=[1, 2])

    # Dump to yaml
    if self.verbose:
      print("- Writing results to yaml")
    file_name = f"{self.data_output}/scan_results_{self.column}{self.extra_file_name}.yaml"
    MakeDirectories(file_name)
    print(f"Created {file_name}")
    with open(file_name, 'w') as yaml_file:
      yaml.dump(scan_results, yaml_file, default_flow_style=False)

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [
      f"{self.data_output}/scan_results_{self.column}{self.extra_file_name}.yaml"
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = self.scan_files
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