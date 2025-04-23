import importlib

class MakeBenchmark():

  def __init__(self):
    """
    A class to make the benchmark scenario datasets and configuration file.
    """
    self.name = None
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
    if ".yaml" not in self.name:
      module = importlib.import_module(self.name)
      module_class = getattr(module, self.name)
      benchmark = module_class()
    else:
      from Dim1CfgToBenchmark import Dim1CfgToBenchmark
      benchmark = Dim1CfgToBenchmark()


    # Fit splines
    if ".yaml" in self.name:
      if self.verbose:
        print("- Fitting splines")
      benchmark.FitSplines()

    # Make dataset
    if self.verbose:
      print("- Making the datasets")
    benchmark.MakeDataset()

    # Make config
    if self.verbose:
      print("- Making the config file")
    benchmark.MakeConfig()

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    if ".yaml" in self.name:
      core_name = self.name.split('/')[-1].split('.yaml')[0]
    else:
      core_name = self.name

    outputs.append(f"configs/run/Benchmark_{core_name}.yaml")

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    return inputs

        