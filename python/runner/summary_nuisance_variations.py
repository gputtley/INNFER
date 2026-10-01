import yaml

import numpy as np

from plotting import plot_summary_nuisance_variations

class SummaryNuisanceVariations():

  def __init__(self):
    """
    A template class.
    """
    self.plots_output = "plots/"
    self.parameter_names = []
    self.up_result_names = []
    self.down_result_names = []
    self.nominal_result_names = []
    self.verbose = False
    self.nuisances_per_page = 20

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

    # Open results
    if self.verbose:
      print("- Loading in results")
    up_results = [yaml.load(open(up_result_name, 'r'), Loader=yaml.FullLoader)["crossings"] for up_result_name in self.up_result_names]
    down_results = [yaml.load(open(down_result_name, 'r'), Loader=yaml.FullLoader)["crossings"] for down_result_name in self.down_result_names]
    nominal_results = [yaml.load(open(nominal_result_name, 'r'), Loader=yaml.FullLoader)["crossings"] for nominal_result_name in self.nominal_result_names]

    # Sort parameters and results by smallest average distance between 0 and 1 and -1 and 0 from the nominal results (list of dictionaries)
    avg_distances = [(abs(nominal_results[i][1] - nominal_results[i][0]) + abs(nominal_results[i][-1] - nominal_results[i][0]))/2 for i in range(len(nominal_results))]
    sorted_indices = np.argsort(avg_distances)
    sorted_parameter_names = [self.parameter_names[i] for i in sorted_indices]
    sorted_up_results = [up_results[i] for i in sorted_indices]
    sorted_down_results = [down_results[i] for i in sorted_indices]
    sorted_nominal_results = [nominal_results[i] for i in sorted_indices]

    pages = len(sorted_parameter_names) // self.nuisances_per_page + (1 if len(sorted_parameter_names) % self.nuisances_per_page != 0 else 0)

    for page in range(pages):
      start_index = page * self.nuisances_per_page
      end_index = (page + 1) * self.nuisances_per_page
      plot_summary_nuisance_variations(
        sorted_parameter_names[start_index:end_index],
        {"Up": sorted_up_results[start_index:end_index], "Down": sorted_down_results[start_index:end_index], "Nominal": sorted_nominal_results[start_index:end_index]},
        show2sigma = True,
        plot_name = f"{self.plots_output}/summary_nuisance_variations_page{page}"
      )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    pages = len(self.parameter_names) // self.nuisances_per_page + (1 if len(self.parameter_names) % self.nuisances_per_page != 0 else 0)
    for page in range(pages):
      outputs.append(f"{self.plots_output}/summary_nuisance_variations_page{page}.pdf")

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    inputs += self.up_result_names
    inputs += self.down_result_names
    inputs += self.nominal_result_names

    return inputs
