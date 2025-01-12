import os

from useful_functions import MakeDirectories

class SummaryAllButOneCollect():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.val_info = None
    self.summary_from = None
    self.val_ind = None

    self.data_input = "data"
    self.data_output = "data"
    self.verbose = True

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def _get_files(self):

    files = []
    for col in self.val_info['row'].columns:
      file_loc = f"{self.data_input}_floating_only_{col}/{self.summary_from.lower()}_results_{col}_{self.val_ind}.yaml"
      files.append(file_loc)

    return files

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    MakeDirectories(self.data_output)
    for file_name in self._get_files():
      if self.verbose:
        print(f"- Coping {file_name} to {self.data_output}")
      os.system(f"cp {file_name} {self.data_output}")

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    for file_name in self._get_files():
      outputs.append(f"{self.data_output}/{file_name.split('/')[-1]}")
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = self._get_files()
    return inputs
