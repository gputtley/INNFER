import yaml

from useful_functions import LoadConfig, MakeDirectories

class SetupDensityFromBenchmark():

  def __init__(self):
    """
    A class to setup the density from the benchamrk scenarios.
    """
    self.cfg = None
    self.file_name = None
    self.benchmark = None
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
    # Load the config
    cfg = LoadConfig(self.cfg)

    # Make the output files
    model_output_name = f"{self.data_output}/{self.file_name}"
    architecture = {"type" : "Benchmark", "benchmark" : self.benchmark}
    MakeDirectories(model_output_name)

    # dump architecture
    architecture_file = f"{model_output_name}_architecture.yaml"
    with open(architecture_file, 'w') as file:
      yaml.dump(architecture, file)

    # Make dummy weights file txt file
    dummy_files = f"{model_output_name}.h5"
    dummy_file = open(dummy_files, 'w')
    dummy_file.write("")
    dummy_file.close()


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []

    model_output_name = f"{self.data_output}/{self.file_name}"
    outputs += [f"{model_output_name}_architecture.yaml"]
    outputs += [f"{model_output_name}.h5"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [self.cfg]

    return inputs

        