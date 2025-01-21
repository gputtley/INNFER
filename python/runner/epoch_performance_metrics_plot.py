import yaml

import numpy as np

from plotting import plot_histograms
from useful_functions import GetDictionaryEntry, GetDictionaryEntryFromYaml, FindKeysAndValuesInDictionaries

class EpochPerformanceMetricsPlot():

  def __init__(self):
    """
    A class to plot the performance metrics of the model as a function of epoch.
    """
    # Default values - these will be set by the configure function
    self.architecture = None
    self.data_input = "data/"
    self.plots_output = "plots"
    self.merged_plot = None

  def _GetKeys(self, dictionary, keys=None):

    if keys is None:
      keys = []

    for key, value in dictionary.items():
      if isinstance(value, dict):
        self._GetKeys(value, keys)
      else:
        keys.append(key)

    return keys

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

    # Get number of epochs
    epochs = GetDictionaryEntryFromYaml(self.architecture, ["epochs"])

    # initialise performance metrics
    performance_metrics = {}

    # loop through epochs and get performance metrics
    first = True
    for epoch in range(epochs+1):

      # open yaml file
      with open(f"{self.data_input}/metrics_epoch_{epoch}.yaml", 'r') as file:
        model_pf = yaml.safe_load(file)

      # if first epoch, get the flattened keys in a list of lists
      if first:
        first = False
        keys, _ = FindKeysAndValuesInDictionaries(model_pf)
        for key in keys:
          performance_metrics['.'.join(key)] = []

      # loop through keys and get values
      for key in keys:
        performance_metrics['.'.join(key)].append(GetDictionaryEntry(model_pf,key))

    # plot all performance metrics
    if self.merged_plot is None:

      for pm in performance_metrics.keys():
        plot_histograms(
          np.array(range(epoch+1)),
          [np.array(performance_metrics[pm])],
          [None],
          name = f"{self.plots_output}/epoch_pm_{pm.replace('.','_')}",
          x_label = "Epoch",
          y_label = pm,      
        )
        plot_histograms(
          np.array(range(1,epoch+1)),
          [np.array(performance_metrics[pm])[1:]],
          [None],
          name = f"{self.plots_output}/epoch_pm_{pm.replace('.','_')}_no_zero",
          x_label = "Epoch",
          y_label = pm,      
        )

    # plot merged performance metrics
    else:

      plot_histograms(
        np.array(range(epoch+1)),
        [(np.array(performance_metrics[pm]) - np.min(performance_metrics[pm]))/(np.max(performance_metrics[pm]) - np.min(performance_metrics[pm])) for pm in self.merged_plot], # standardise
        self.merged_plot,
        name = f"{self.plots_output}/epoch_pm_merged",
        x_label = "Epoch",
        y_label = "$(x-x_{min})/(x_{max}-x_{min})$",      
      )
      plot_histograms(
        np.array(range(1,epoch+1)),
        [(np.array(performance_metrics[pm])[1:] - np.min(performance_metrics[pm][1:]))/(np.max(performance_metrics[pm][1:]) - np.min(performance_metrics[pm][1:])) for pm in self.merged_plot], # standardise
        self.merged_plot,
        name = f"{self.plots_output}/epoch_pm_merged_no_zero",
        x_label = "Epoch",
        y_label = "$(x-x_{min})/(x_{max}-x_{min})$",      
      )



  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    if self.merged_plot is None:
      for pm in self.merged_plot:
        outputs.append(f"{self.plots_output}/epoch_pm_{pm.replace('.','_')}.pdf")
        outputs.append(f"{self.plots_output}/epoch_pm_{pm.replace('.','_')}_no_zero.pdf")
    else:
      outputs.append(f"{self.plots_output}/epoch_pm_merged.pdf")
      outputs.append(f"{self.plots_output}/epoch_pm_merged_no_zero.pdf")
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    for epoch in range(GetDictionaryEntryFromYaml(self.architecture, ["epochs"])+1):
      inputs.append(f"{self.data_input}/metrics_epoch_{epoch}.yaml")
    return inputs

        