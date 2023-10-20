import copy
import numpy as np
from plotting import plot_histograms

class DataProcessor():

  def __init__(self, data):

    self.data = data

    # Check inputs
    self._data_keys = ["train","test","val"]
    for k in data.keys():
      if k not in self._data_keys:
        print(f"ERROR: Invalid key {k}.")
    if "train" not in data.keys():
      print("ERROR: train not in keys.")

    self.plot_dir = "plots"
    self.standardise_data = {"X":False,"Y":False}
    self.standardisation_mean = {"X":None,"Y":None}
    self.standardisation_std = {"X":None,"Y":None}
    
    self.add_dummy_X = False

  def Standardise(self, data, feature_type):
    # Calculate standardisation
    if not (isinstance(self.standardisation_mean[feature_type], np.ndarray) or isinstance(self.standardisation_std[feature_type], np.ndarray)):
      self.standardisation_mean[feature_type] = np.mean(data["train"][feature_type], axis=0)
      self.standardisation_std[feature_type] = np.std(data["train"][feature_type], axis=0)
      self.standardisation_std[feature_type][self.standardisation_std[feature_type] == 0] = 1
    # Standardise data
    for dk in self.data.keys():
      data[dk][feature_type] = (data[dk][feature_type]-self.standardisation_mean[feature_type])/self.standardisation_std[feature_type]   
    return data

  def UnStandardise(self, input, feature_type="X"):
    # UnStandardise data
    output = (input*self.standardisation_std[feature_type])  + self.standardisation_mean[feature_type]
    return output

  def AddDummyX(self, data):
    for dk in self.data.keys():
      data[dk]["X"] = np.column_stack((data[dk]["X"].flatten(), np.random.normal(0.0, 1.0, (len(data[dk]["X"].flatten()),))))
    return data

  def PreProcess(self, purpose=None, only_context=False):

    # Data to return, this means the initial data is unchanged by the class
    data = copy.deepcopy(self.data)
    
    # Fix for 1D latent space
    if self.add_dummy_X and not only_context:
      data = self.AddDummyX(data)

    # Standardise
    for feature_type in ["X","Y"]:
      if only_context and feature_type == "X": continue
      if self.standardise_data[feature_type]:
        data = self.Standardise(data, feature_type)

    # Get naming correct for BayesFlow
    for dk in self.data.keys():
      if purpose == "training":
        data[dk]["sim_data"] = data[dk]["Y"]
        if not only_context:
          data[dk]["prior_draws"] = data[dk]["X"]
        if "X" in data[dk].keys(): del data[dk]["X"]
        if "Y" in data[dk].keys(): del data[dk]["Y"]
      elif purpose == "inference":
        data[dk]["direct_conditions"] = data[dk]["Y"]
        if not only_context:
          data[dk]["parameters"] = data[dk]["X"]
        if "X" in data[dk].keys(): del data[dk]["X"]
        if "Y" in data[dk].keys(): del data[dk]["Y"]
    return data

  def PostProcessProb(self,prob):
    prob = prob/np.prod(self.standardisation_std["X"])
    return prob

  def PlotUniqueY(self, n_bins=40, data_key="val"):

    # Plot each column separately
    for col in range(self.data[data_key]["X"].shape[1]):

      # Find combined binning first
      _, bins = np.histogram(self.data[data_key]["X"][:,col], bins=n_bins)

      # Loop through unique bins
      hists = []
      hist_names = []
      unique_rows = np.unique(self.data[data_key]["Y"], axis=0)
      for ur in unique_rows:
        matching_rows = np.all(self.data[data_key]["Y"] == ur, axis=1)
        X_cut = self.data[data_key]["X"][matching_rows]
        h, _ = np.histogram(X_cut, bins=bins)
        hists.append(h)
        if self.data[data_key]["Y"].shape[1] > 1:
          hist_names.append("y=({})".format(",".join([str(i) for i in ur])))
        else:
          hist_names.append("y={}".format(ur[0]))

      # Plot histograms
      plot_name = self.plot_dir+f"/initial_distributions_x{col}"
      plot_histograms(
        bins[:-1],
        hists,
        hist_names,
        title_right = "",
        name = plot_name,
        x_label = f"x[{col}]",
        y_label = "Events",
      )