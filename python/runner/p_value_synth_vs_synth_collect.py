import yaml

from useful_functions import MakeDirectories, FindKeysAndValuesInDictionaries, GetDictionaryEntry

class PValueSynthVsSynthCollect():

  def __init__(self):
    """
    A template class.
    """

    self.data_input = "data/"
    self.data_output = "data/"
    self.number_of_toys = None
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

    first = True
    for i in range(self.number_of_toys):
      with open(f"{self.data_input}/p_value_dataset_comparison_{self.sim_type}_{i}.yaml", 'r') as yaml_file:
        metrics = yaml.load(yaml_file, Loader=yaml.FullLoader)

      if first:
        first = False
        keys, _ = FindKeysAndValuesInDictionaries(metrics)
        for key in keys:
          out['.'.join(key)] = {}

      # loop through keys and get values
      for key in keys:
        out['.'.join(key)][i] = GetDictionaryEntry(metrics,key)


    # Write out the results to yaml
    if self.verbose:
      print("- Writing the results to yaml")
    MakeDirectories(self.data_output)
    with open(f"{self.data_output}/p_value_dataset_comparison_{self.sim_type}.yaml", 'w') as file:
      yaml.dump(out, file)

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    #outputs = []
    outputs = [f"{self.data_output}/p_value_dataset_comparison_{self.sim_type}.yaml"]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for i in range(self.number_of_toys):
      inputs.append(f"{self.data_input}/p_value_dataset_comparison_{self.sim_type}_{i}.yaml")
    return inputs

        