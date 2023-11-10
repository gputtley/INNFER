from data_loader import DataLoader
from innfer_trainer import InnferTrainer
from preprocess import PreProcess
from plotting import plot_histograms, plot_histogram_with_ratio
import bayesflow as bf
import tensorflow as tf
import pandas as pd
import numpy as np
import copy
import gc

class Validation():
  """
  Validation class for evaluating and visualizing the performance of conditional invertible neural network models. 
  """
  def __init__(self, model, options={}):
    """
    Initialize the Validation class.

    Args:
        model: Trained Bayesian neural network model.
        options (dict): Dictionary of options for the validation process.
    """
    self.model = model
    self.data_parameters = {}
    self.data_dir = "./data/"
    self.plot_dir = "./plots/"
    self._SetOptions(options)

    self.pp = PreProcess()
    self.pp.parameters = self.data_parameters
    self.pp.output_dir = self.data_dir

  def _SetOptions(self, options):
    """
    Set options for the validation process.

    Args:
        options (dict): Dictionary of options for the validation process.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def PlotGeneration(self, row, columns=None, data_key="val", n_bins=40, ignore_quantile=0.01):
    """
    Plot generation comparison between simulated and synthetic data for a given row.

    Args:
        row (list): List representing the unique row for comparison.
        columns (list): List of column names for plotting. If None, uses Y_columns from data_parameters.
        data_key (str): Key specifying the dataset (e.g., "val" for validation data).
        n_bins (int): Number of bins for histogram plotting.
        ignore_quantile (float): Fraction of data to ignore from both ends during histogram plotting.
    """
    X, Y, wt = self.pp.LoadSplitData(dataset=data_key, get=["X","Y","wt"])
    
    if columns is None:
      columns = self.data_parameters["Y_columns"]

    row = np.array(list(row))

    matching_rows = np.all(Y.to_numpy() == row, axis=1)
    X = X.to_numpy()[matching_rows]
    wt = wt.to_numpy()[matching_rows]
    del Y
    gc.collect()

    synth = self.model.Sample(np.array([list(row)]), columns=columns)

    for col in range(X.shape[1]):

      trimmed_X = X[:,col]
      lower_value = np.quantile(trimmed_X, ignore_quantile)
      upper_value = np.quantile(trimmed_X, 1-ignore_quantile)
      trimmed_indices = ((trimmed_X >= lower_value) & (trimmed_X <= upper_value))
      trimmed_X = trimmed_X[trimmed_indices]
      trimmed_wt = wt[trimmed_indices].flatten()

      sim_hist, bins  = np.histogram(trimmed_X, weights=trimmed_wt,bins=n_bins)
      sim_hist_err_sq, _  = np.histogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      synth_hist, _  = np.histogram(synth[:,col], bins=bins)
      
      file_extra_name = self._GetYName(row, purpose="file")
      plot_extra_name = self._GetYName(row, purpose="plot")

      plot_histogram_with_ratio(
        sim_hist, 
        synth_hist, 
        bins, 
        name_1='Simulated', 
        name_2='Synthetic',
        xlabel=self.data_parameters["X_columns"][col],
        name=f"{self.plot_dir}/generation_{self.data_parameters['X_columns'][col]}_y_{file_extra_name}", 
        title_right = f"y={plot_extra_name}",
        density = True,
        use_stat_err = False,
        errors_1=np.sqrt(sim_hist_err_sq), 
        errors_2=np.sqrt(synth_hist),
        )

  def _GetYName(self, ur, purpose="plot"):
    """
    Get a formatted label for a given unique row.

    Args:
        ur (list): List representing the unique row.
        purpose (str): Purpose of the label, either "plot" or "file
    """
    label_list = [str(i) for i in ur] 
    if purpose == "file":
      name = "_".join([i.replace(".","p").replace("-","m") for i in label_list])
    elif purpose == "plot":
      if len(label_list) > 1:
        name = "({})".format(",".join(label_list))
      else:
        name = label_list[0]
    return name