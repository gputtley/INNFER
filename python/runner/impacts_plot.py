import yaml

from plotting import plot_pulls_and_impacts
from useful_functions import LoadConfig, Translate

class ImpactsPlot():

  def __init__(self):
    """
    A class to plot the pulls and impacts of different systematics on measurements.
    """
    self.cfg = None
    self.impacts_input = "data/impacts.yaml"
    self.pulls_input = "data/"
    self.plots_output = "plots/"
    self.extra_input_name = ""
    self.impacts_per_page = 20
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
    # Load config file
    cfg = LoadConfig(self.cfg)

    # Open the impacts file
    with open(self.impacts_input, 'r') as f:
      impacts_data = yaml.safe_load(f)

    # Open the POI file
    poi_file = f"{self.pulls_input}/covariance_results_{impacts_data['poi']}{self.extra_input_name}.yaml"
    with open(poi_file, 'r') as f:
      poi_data = yaml.safe_load(f)

    # Define dictionary to store all info
    pulls_and_impacts = []

    # Loop over impacts
    for k, v in impacts_data["impacts"].items():

      # Add impacts
      pulls_and_impacts.append({"parameter": k, "impacts" : [-v, v] if isinstance(v, float) else [v[-1],v[-1]]})

      # Open pulls
      pulls_file = f"{self.pulls_input}/covariance_results_{k}{self.extra_input_name}.yaml"
      with open(pulls_file, 'r') as f:
        pulls_data = yaml.safe_load(f)

      # Add pulls
      pulls_and_impacts[-1]["pulls"] = pulls_data["crossings"]

      # Check if constrained parameter
      pulls_and_impacts[-1]["constrained_parameter"] = False
      if "nuisance_constraints" in cfg["inference"].keys():
        if k in cfg["inference"]["nuisance_constraints"]:
          pulls_and_impacts[-1]["constrained_parameter"] = True

    # Sort by impacts absolute value (biggest first)
    pulls_and_impacts = sorted(pulls_and_impacts, key=lambda x: max(abs(x["impacts"][0]), abs(x["impacts"][1])), reverse=True)

    # Now plot the pulls and impacts
    pages = [pulls_and_impacts[i:i + self.impacts_per_page] for i in range(0, len(pulls_and_impacts), self.impacts_per_page)]
    for page_ind, page in enumerate(pages):
      plot_name = f"{self.plots_output}/impacts{self.extra_input_name}_page{page_ind+1}"
      plot_pulls_and_impacts(page, plot_name, poi_name=Translate(impacts_data['poi']), poi_crossings=poi_data["crossings"])


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Define outputs
    outputs = [f"{self.plots_output}/impacts{self.extra_input_name}_page{page_ind+1}.pdf" for page_ind in range((len(LoadConfig(self.cfg)["nuisances"]) + self.impacts_per_page - 1) // self.impacts_per_page)]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [self.cfg, self.impacts_input]

    # Load config
    cfg = LoadConfig(self.cfg)

    # Loop through parameters
    for par in cfg["pois"] + cfg["nuisances"]:
      pulls_file = f"{self.pulls_input}/covariance_results_{par}{self.extra_input_name}.yaml"
      inputs.append(pulls_file)

    return inputs

        