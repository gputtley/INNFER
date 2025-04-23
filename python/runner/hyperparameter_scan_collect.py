import copy
import os
import yaml

class HyperparameterScanCollect():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.save_extra_names = []

    # Required input which can just be parsed
    self.metric = ""
    self.file_name = None

    # other
    self.data_input = "data/"
    self.data_output = "data/"
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

    if self.verbose:
      print("- Finding best performing model")

    best_metric_val = None
    best_metric_extra_name = None

    for save_extra_name in self.save_extra_names:

      # Open metrics
      metric_name = f"{self.data_input}/metrics{save_extra_name}.yaml"
      with open(metric_name, 'r') as yaml_file:
        metric = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Load info in
      metric_name, min_max = self.metric.split(",")
      for k in metric_name.split(":"):
        metric = metric[k]

      # Get best model
      if (best_metric_val is None) or (min_max == "max" and metric > best_metric_val) or (min_max == "min" and metric < best_metric_val):
        best_metric_val = copy.deepcopy(metric)
        best_metric_extra_name = copy.deepcopy(save_extra_name)

    if self.verbose:
      print(f"- Best model architecture is {self.data_input}/{self.file_name}{best_metric_extra_name}_architecture.yaml")
      print("- Copying over best performing model")

    # Copy over best model
    os.system(f"cp {self.data_input}/{self.file_name}{save_extra_name}.h5 {self.data_output}/{self.file_name}.h5")
    os.system(f"cp {self.data_input}/{self.file_name}{save_extra_name}_architecture.yaml {self.data_output}/{self.file_name}_architecture.yaml")

  def Outputs(self):
    """
    Return a list of outputs given by class
    """

    outputs = [
      f"{self.data_output}/{self.file_name}.h5",
      f"{self.data_output}/{self.file_name}_architecture.yaml",
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    inputs += [f"{self.data_input}/metrics{save_extra_name}.yaml" for save_extra_name in self.save_extra_names]
    inputs += [f"{self.data_input}/{self.file_name}{save_extra_name}.h5" for save_extra_name in self.save_extra_names]
    inputs += [f"{self.data_input}/{self.file_name}{save_extra_name}_architecture.yaml" for save_extra_name in self.save_extra_names]
    return inputs
