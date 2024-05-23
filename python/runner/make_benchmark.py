class MakeBenchmark():

  def __init__(self):
    """
    A class to make the benchmark scenario datasets and configuration file.
    """
    self.name = None

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

    from benchmarks import Benchmarks
    benchmark = Benchmarks(name=self.name)
    # Fit splines
    if ".yaml" in self.name:
      benchmark.FitSplines()
    # Make dataset and config
    benchmark.MakeDataset()
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

    from benchmarks import Benchmarks
    benchmark = Benchmarks(name=self.name)
    cfg = benchmark.MakeConfig(return_cfg=True)
    for _, file_name in cfg["files"].items():
      outputs.append(file_name)

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
        