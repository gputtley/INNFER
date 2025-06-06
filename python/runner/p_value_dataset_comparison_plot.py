import yaml

import numpy as np

from plotting import plot_histograms
from useful_functions import FindKeysAndValuesInDictionaries

class PValueDatasetComparisonPlot():

  def __init__(self):
    """
    A template class.
    """
    self.synth_vs_synth_input = "data/"
    self.sim_vs_synth_input = "data/"
    self.plots_output = "plots/"
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
    # Open results
    if self.verbose:
      print("- Loading in the sim vs syth results")
    with open(f"{self.sim_vs_synth_input}/metrics.yaml", 'r') as yaml_file:
      sim_vs_synth_results = yaml.load(yaml_file, Loader=yaml.FullLoader)
    if self.verbose:
      print("- Loading in the synth vs syth results")
    with open(f"{self.synth_vs_synth_input}/metrics.yaml", 'r') as yaml_file:
      synth_vs_synth_results = yaml.load(yaml_file, Loader=yaml.FullLoader)

    keys, vals = FindKeysAndValuesInDictionaries(sim_vs_synth_results)

    for key, val in zip(keys,vals):

      metric = '.'.join(key)

      # Make histogram of values
      if self.verbose:
        print(f"- Making the histogram for {metric}")
      hist, bins = np.histogram(list(synth_vs_synth_results[metric].values()), bins=20)

      # Add extra zero bin either side
      hist = np.append([0], hist)
      bins = np.append([(2*bins[0]) - bins[1]], bins)
      hist = np.append(hist, [0])
      bins = np.append(bins, [(2*bins[-1]) - bins[-2]])

      # Add extra hist entry
      hist = np.append(hist, [0])

      # Add bins to syth vs sim value
      if val < bins[0]:
        bins = np.append([val], bins)
        hist = np.append([0], hist)
      elif val > bins[-1]:
        bins = np.append(bins, [val])
        hist = np.append(hist, [0])

      # Get filled bins
      fill_hist = np.append(0.0, hist[bins>val])
      fill_bins = np.append(val, bins[bins>val])

      # get p-values
      all_results = np.array(list(synth_vs_synth_results[metric].values()))
      p_value = len(all_results[all_results>val]) / len(all_results)

      # plot the histogram
      if self.verbose:
        print(f"- Plotting the histogram for {metric}")
      plot_histograms(
        bins,
        [hist],
        [f"{len(all_results)} Synth Vs Synth Bootstraps"],
        drawstyle = "steps-pre",
        name = f"{self.plots_output}/p_value_dataset_comparison_{metric.replace('.','_')}",
        x_label = metric,
        y_label = "Count",
        anchor_y_at_0 = True,
        vertical_lines = [val],
        vertical_line_names = ["Sim vs Synth","Null Hypothesis"],
        vertical_line_colors=["orange"],
        fill_between_bins = fill_bins,
        fill_between_hist = fill_hist,
        fill_between_step = "pre",
        fill_between_color = "blue",
        fill_between_alpha = 0.3,
        title_right = f"p-value = {round(p_value,2)}"
      )

    # Make dummy output empty txt file
    with open(f"{self.plots_output}/dummy.txt", 'w') as file:
      file.write("")


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [f"{self.plots_output}/dummy.txt"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [
      f"{self.sim_vs_synth_input}/metrics.yaml",
      f"{self.synth_vs_synth_input}/metrics.yaml"
    ]
    return inputs

        