import yaml

from useful_functions import MakeDirectories

class C2STCollect():

  def __init__(self):
    """
    A template class.
    """

    self.data_input = "data/"
    self.data_output = "data/"
    self.number_of_c2st = None
    self.sim_type = "val"
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
    # Put inputs in one dictionary
    out = {}
    if self.verbose:
      print("- Loading in the inputs")

    for metric in ["auc","accuracy"]:
      out[metric] = {}
      for i in range(self.number_of_c2st):
        with open(f"{self.data_input}/c2st_{self.sim_type}_{i}.yaml", 'r') as yaml_file:
          c2st = yaml.load(yaml_file, Loader=yaml.FullLoader)
          for k, v in c2st[metric].items():
            out[metric][k] = v

    # Write out the results to yaml
    if self.verbose:
      print("- Writing the results to yaml")
    MakeDirectories(self.data_output)
    with open(f"{self.data_output}/c2st_{self.sim_type}.yaml", 'w') as file:
      yaml.dump(out, file)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [f"{self.data_output}/c2st_{self.sim_type}.yaml"]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for i in range(self.number_of_c2st):
      inputs.append(f"{self.data_input}/c2st_{self.sim_type}_{i}.yaml")
    return inputs

        