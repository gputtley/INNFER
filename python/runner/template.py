# This is a dummy template class

class Template():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.var = None

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
    # Code the run the class

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
    inputs = []
    return inputs

  def CheckInputs(self):
    """
    Check that all the relevant inputs have been parsed to the class
    """
    for var in self.inputs:
      if getattr(self, var) == None:
        raise AttributeError(f"'{var}' variable required by class.")
        