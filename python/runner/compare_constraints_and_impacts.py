import yaml

from plotting import plot_barred_comparison
from useful_functions import LoadConfig

class CompareConstraintsAndImpacts():

  def __init__(self):
    """
    A class to plot the pulls and impacts of different systematics on measurements.
    """
    self.cfg = None
    self.impacts_input = []
    self.pulls_input = []
    self.plots_output = "plots/"
    self.extra_input_name = ""
    self.number_of_parameters = 18
    self.verbose = False
    self.names = None

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

    columns = None
    scale_constraints = []
    constraints = {}
    impacts = {}

    for ind in range(len(self.impacts_input)):

      # Open the impacts file
      with open(self.impacts_input[ind], 'r') as f:
        impacts_data = yaml.safe_load(f)

      # Define dictionary to store all info
      name = self.names[ind] if self.names is not None else f"{ind}"
      constraints[name] = []
      impacts[name] = []

      if columns is None:
        columns = impacts_data["impacts"].keys()
        all_impacts = [abs(impacts_data["impacts"][k]) for k in columns]
        # sort columns decreasing from max impact
        columns = [x for _, x in sorted(zip(all_impacts, columns), key=lambda pair: pair[0], reverse=True)][:self.number_of_parameters]

      # Loop over impacts
      for k in columns:

        v = impacts_data["impacts"][k]
        impacts[name].append(abs(v))

        # Open pulls
        pulls_file = f"{self.pulls_input[ind]}/covariance_results_{k}{self.extra_input_name}.yaml"
        with open(pulls_file, 'r') as f:
          pulls_data = yaml.safe_load(f)

        up_uncert = abs(pulls_data["crossings"][1] - pulls_data["crossings"][0])
        down_uncert = abs(pulls_data["crossings"][0] - pulls_data["crossings"][-1])
        ave_uncert = (up_uncert + down_uncert)/2

        constraints[name].append(ave_uncert)


    # If it is not a constrained nuisance then scale the max constraint to 1
    for col in columns:
      if col not in cfg["inference"]["nuisance_constraints"]:
        max_constraint = max([constraints[name][columns.index(col)] for name in constraints])
        scale_constraints.append(max_constraint)
        for name in constraints.keys():
          constraints[name][columns.index(col)] /= max_constraint
      else:
        scale_constraints.append(1.0)

    plot_barred_comparison(
      constraints,
      columns,
      scale_values = scale_constraints,
      y_label = "Constraints",
      name = f"{self.plots_output}/constraints_comparison",
    )


    plot_barred_comparison(
      impacts,
      columns,
      y_label = "Impacts",
      name = f"{self.plots_output}/impacts_comparison",
    )




  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Define outputs
    outputs = []
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [self.cfg] + self.impacts_input

    # Load config
    cfg = LoadConfig(self.cfg)

    # Loop through parameters
    for par in cfg["pois"] + cfg["nuisances"]:
      for pull_input in self.pulls_input:
        pulls_file = f"{pull_input}/covariance_results_{par}{self.extra_input_name}.yaml"
        inputs.append(pulls_file)

    return inputs

        