import yaml

import numpy as np

from useful_functions import LoadConfig, MakeDirectories

class ApproximateImpacts():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.data_input = None
    self.data_output = None
    self.cfg = None

    self.impact_to = None
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
    
    # Open the data input covariance yaml file
    with open(self.data_input, 'r') as f:
      data = yaml.safe_load(f)

    # Load the config
    cfg = LoadConfig(self.cfg)

    # Get the poi
    poi = self.impact_to if self.impact_to is not None else cfg["pois"][0]
    poi_index = data["columns"].index(poi)

    if self.verbose:
      print(f"- Calculating approximate impacts to {poi}")

    impacts = {}
    for ind, var in enumerate(data["columns"]):
      if var == poi: continue
      impacts[var] = float(data["covariance"][poi_index][ind] / np.sqrt(data["covariance"][ind][ind])) if data["covariance"][ind][ind] > 0 else 0.0

    # Sort the impacts by absolute value
    impacts = dict(sorted(impacts.items(), key=lambda item: abs(item[1]), reverse=True))

    # print the impacts
    if self.verbose:
      print("- Approximate impacts:")
      for ind, (var, impact) in enumerate(impacts.items()):
        print(f"  - Rank{ind+1} : {var} : {impact}")

    # Save the impacts to yaml
    MakeDirectories(self.data_output)
    with open(self.data_output, 'w') as f:
      yaml.dump(impacts, f)
    

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
    inputs = [self.data_input, self.cfg]
    return inputs

        