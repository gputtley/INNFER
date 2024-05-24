class PreProcess():

  def __init__(self):
    """
    A class to preprocess the datasets and produce the data 
    parameters yaml file as well as the train, test and 
    validation datasets.
    """
    self.cfg = cfg

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
    self.CheckInputs()

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [
      "name",
    ]
    return inputs

  def CheckInputs(self):
    """
    Check that all the relevant inputs have been parsed to the class
    """
    for var in self.Inputs():
      if getattr(self, var) == None:
        raise AttributeError(f"'{var}' variable required by class.")
        