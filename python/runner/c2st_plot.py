import yaml

import numpy as np

from plotting import plot_histograms

class C2STPlot():

  def __init__(self):
    """
    A template class.
    """
    self.data_input = "data/"
    self.plots_output = "plots/"
    self.sim_type = "val"
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
    # Open parameters
    if self.verbose:
      print("- Loading in the results")
    with open(f"{self.data_input}/c2st_{self.sim_type}.yaml", 'r') as yaml_file:
      results = yaml.load(yaml_file, Loader=yaml.FullLoader)

    for metric in ["auc","accuracy"]:

      # Make histogram of values
      if self.verbose:
        print(f"- Making the histogram for {metric}")
      hist, bins = np.histogram(list(results[metric].values()), bins=20)

      # Add extra zero bin either side
      hist = np.append([0], hist)
      bins = np.append([(2*bins[0]) - bins[1]], bins)
      hist = np.append(hist, [0])
      bins = np.append(bins, [(2*bins[-1]) - bins[-2]])

      # Add bins to 0.5
      if 0.5 < bins[0]:
        bins = np.append([0.5], bins)
        hist = np.append([0], hist)
      elif 0.5 > bins[-1]:
        bins = np.append(bins, [0.5])
        hist = np.append(hist, [0])

      if metric == "auc":
        metric_name = "ROC AUC"
      elif metric == "accuracy":
        metric_name = "Accuracy"

      # plot the histogram
      if self.verbose:
        print(f"- Plotting the histogram for {metric}")
      plot_histograms(
        bins[:-1],
        [hist],
        ["Tests"],
        drawstyle = "steps-post",
        name = f"{self.plots_output}/c2st_{metric}_{self.sim_type}",
        x_label = metric_name,
        y_label = "Count",
        anchor_y_at_0 = True,
        vertical_lines = [float(np.mean(list(results[metric].values()))), 0.5],
        vertical_line_names = ["Mean Test","Random Guess"],
        vertical_line_colors=["orange","red"],
      )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    for metric in ["auc","accuracy"]:
      outputs.append(f"{self.plots_output}/c2st_{metric}_{self.sim_type}.pdf")
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [f"{self.data_input}/c2st_{self.sim_type}.yaml"]
    return inputs

        