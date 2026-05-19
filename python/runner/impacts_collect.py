import yaml

from useful_functions import LoadConfig, MakeDirectories

class ImpactsCollect():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.best_fit_input = None
    self.impacts_input = None
    self.data_output = None
    self.impact_to = None
    self.cfg = None
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

    if self.impact_to is None:
      print("- No impact_to specified, using first poi in config")
      cfg = LoadConfig(self.cfg)
      self.impact_to = cfg["pois"][0]

    # Code the run the class
    cfg = LoadConfig(self.cfg)

    # Load the best fit yaml
    with open(self.best_fit_input, 'r') as f:
      best_fit = yaml.safe_load(f)
    col_index = best_fit['columns'].index(self.impact_to)
    best_fit_val = best_fit['best_fit'][col_index]

    # Load the impacts
    impacts = {"poi" : self.impact_to, "impacts" : {}}
    for v in self.impacts_input:
      parameter = v["parameter"]
      shift = v["shift"]
      file = v["file"]
      if parameter == self.impact_to:
        continue

      if parameter not in impacts["impacts"]:
        impacts["impacts"][parameter] = [0.0,0.0]
      with open(file, 'r') as f:
        impacts_file = yaml.safe_load(f)
      col_index = impacts_file['columns'].index(self.impact_to)
      value = impacts_file['best_fit'][col_index] - best_fit_val
      if shift == "down":
        impacts["impacts"][parameter][0] = value
      elif shift == "up":
        impacts["impacts"][parameter][1] = value

    # Sort the impacts by absolute value
    impacts["impacts"] = dict(sorted(impacts["impacts"].items(), key=lambda item: abs(item[1][1]), reverse=True))

    # print the impacts
    if self.verbose:
      print("- Approximate impacts:")
      for ind, (var, impact) in enumerate(impacts["impacts"].items()):
        print(f"  - Rank{ind+1} : {var} : {impact}")

    # Save the impacts 
    MakeDirectories(self.data_output)
    with open(self.data_output, 'w') as f:
      yaml.dump(impacts, f)
    if self.verbose:
      print(f"- Saved impacts to {self.data_output}")

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
    inputs = [self.best_fit_input]

    for v in self.impacts_input:
      inputs.append(v["file"])

    return inputs

        