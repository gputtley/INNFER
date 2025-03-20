import numpy as np
import pandas as pd

from functools import partial
from itertools import product
from scipy.stats import wasserstein_distance

from data_processor import DataProcessor
from useful_functions import GetYName, MakeDictionaryEntry

class HistogramMetrics():

  def __init__(self, network, parameters, sim_files):
    self.network = network
    self.parameters = parameters
    self.sim_files = sim_files

    self.hists_exist = False
    self.sim_hists = None
    self.sim_hist_uncerts = None
    self.synth_hists = None
    self.synth_hist_uncerts = None

  def MakeHistograms(self, n_samples=100000, n_bins=40):

    self.sim_hists = {k:{} for k in self.sim_files.keys()}
    self.sim_hist_uncerts = {k:{} for k in self.sim_files.keys()}
    self.synth_hists = {k:{} for k in self.sim_files.keys()}
    self.synth_hist_uncerts = {k:{} for k in self.sim_files.keys()}
    self.bin_centers = {k:{} for k in self.sim_files.keys()}

    # initiate values for loop
    sim_dps = {}

    # loop through the sim files
    first = True
    for sim_file_name, sim_file in self.sim_files.items():

      # make dataprocessor
      sim_dps[sim_file_name] = DataProcessor(
        [sim_file],
        "parquet",
        wt_name = "wt",
        options = {
          "parameters" : self.parameters
        }
      )

      # Get unique Y of dataset
      if len(self.parameters["Y_columns"]) > 0:
        unique_values = sim_dps[sim_file_name].GetFull(method="unique", functions_to_apply=["untransform"])
        if first:
          first = False
          unique_columns = [k for k in unique_values.keys() if k in self.parameters["Y_columns"]]
          unique_y_values = {k : [] for k in unique_columns}
          unique_y_combinations_per_file = {}
        unique_y_values = {k : sorted(list(set(unique_y_values[k] + unique_values[k]))) for k in unique_columns}
        unique_y_values_per_file = {k : sorted(unique_values[k]) for k in unique_columns}
        unique_y_combinations = list(product(*unique_y_values.values()))
        unique_y_combinations_per_file[sim_file_name] = list(product(*unique_y_values_per_file.values()))

    # make all unique combinations
    if len(self.parameters["Y_columns"]) > 0:
      unique_y_combinations = list(product(*unique_y_values.values()))
    else:
      unique_y_combinations = [None]

    # Loop through unique Y
    for uc in unique_y_combinations:

      if uc == None:
        uc_name = "all"
        Y = pd.DataFrame([])
        selection = None
      else:
        uc_name = GetYName(uc, purpose="file")
        Y = pd.DataFrame(np.array([uc]), columns=unique_columns, dtype=np.float64)
        selection = " & ".join([f"({k}=={uc[ind]})" for ind, k in enumerate(unique_columns)])

      # Make synthetic data processors
      synth_dp = DataProcessor(
        [[partial(self.network.Sample, Y)]],
        "generator",
        n_events = n_samples,
        options = {
          "parameters" : self.parameters
        }
      )

      # loop through columns
      for col in self.parameters["X_columns"]:

        # Make synth histograms
        synth_hist, synth_hist_uncert, bins = synth_dp.GetFull(
          method = "histogram_and_uncert",
          bins = n_bins,
          column = col,
          density = True,
          )

        # loop through the sim files
        for sim_file_name, sim_file in self.sim_files.items():

          if uc is None or uc in unique_y_combinations_per_file[sim_file_name]:

            # Make sim histograms            
            sim_hist, sim_hist_uncert, bins = sim_dps[sim_file_name].GetFull(
              method = "histogram_and_uncert",
              functions_to_apply = ["untransform"],
              bins = bins,
              column = col,
              density = True,
              extra_sel = selection
              )


            # Make sure the dictionary is initiated
            if uc_name not in self.sim_hists[sim_file_name].keys():
              self.sim_hists[sim_file_name][uc_name] = {}
              self.sim_hist_uncerts[sim_file_name][uc_name] = {}
            if uc_name not in self.synth_hists[sim_file_name].keys():
              self.synth_hists[sim_file_name][uc_name] = {}
              self.synth_hist_uncerts[sim_file_name][uc_name] = {}
            if uc_name not in self.bin_centers[sim_file_name].keys():
              self.bin_centers[sim_file_name][uc_name] = {}

            # Add the histograms to the dictionary
            self.sim_hists[sim_file_name][uc_name][col] = sim_hist
            self.sim_hist_uncerts[sim_file_name][uc_name][col] = sim_hist_uncert
            self.synth_hists[sim_file_name][uc_name][col] = synth_hist
            self.synth_hist_uncerts[sim_file_name][uc_name][col] = synth_hist_uncert
            self.bin_centers[sim_file_name][uc_name][col] = (bins[:-1] + bins[1:]) / 2

    self.RemoveZeroBins()
    self.hists_exist = True


  def RemoveZeroBins(self):

    for sim_file_name, sim_hist_per_file in self.sim_hists.items():
      for uc_name, sim_hist in sim_hist_per_file.items():
        for col in sim_hist.keys():
          non_zero_sim_indices = np.where((self.sim_hists[sim_file_name][uc_name][col] > 0) & (self.synth_hists[sim_file_name][uc_name][col] > 0))[0]
          self.sim_hists[sim_file_name][uc_name][col] = self.sim_hists[sim_file_name][uc_name][col][non_zero_sim_indices]
          self.sim_hist_uncerts[sim_file_name][uc_name][col] = self.sim_hist_uncerts[sim_file_name][uc_name][col][non_zero_sim_indices]
          self.synth_hists[sim_file_name][uc_name][col] = self.synth_hists[sim_file_name][uc_name][col][non_zero_sim_indices]
          self.synth_hist_uncerts[sim_file_name][uc_name][col] = self.synth_hist_uncerts[sim_file_name][uc_name][col][non_zero_sim_indices]
          self.bin_centers[sim_file_name][uc_name][col] = self.bin_centers[sim_file_name][uc_name][col][non_zero_sim_indices]


  def GetChiSquared(self, add_sum=True, add_mean=True):

    # Make histograms is they don't exist
    if not self.hists_exist:
      self.MakeHistograms()

    # Setup dictionaries
    chi_squared_dict = {}
    dof_dict = {}
    chi_squared_per_dof_dict = {}

    # Loop through the sim files
    for sim_file_name, sim_hist_per_file in self.sim_hists.items():
      total_sim_file_chi_squared_per_dof = 0
      count_sim_file_chi_squared_per_dof = 0
      # Loop through the unique combinations
      for uc_name, sim_hist in sim_hist_per_file.items():
        total_uc_chi_squared_per_dof = 0
        count_uc_chi_squared_per_dof = 0
        # Loop through the columns
        for col in sim_hist.keys():

          # Calculate chi squared
          chi_squared = float(np.sum((self.synth_hists[sim_file_name][uc_name][col]-self.sim_hists[sim_file_name][uc_name][col])**2/(self.synth_hist_uncerts[sim_file_name][uc_name][col]**2 + self.sim_hist_uncerts[sim_file_name][uc_name][col]**2)))
          dof = len(self.synth_hists[sim_file_name][uc_name][col])
          chi_squared_per_dof = chi_squared / dof

          # Add to dictionaries
          chi_squared_dict = MakeDictionaryEntry(chi_squared_dict, [f"chi_squared_{sim_file_name}",uc_name,col], chi_squared)
          dof_dict = MakeDictionaryEntry(dof_dict, [f"dof_for_chi_squared_{sim_file_name}",uc_name,col], dof)
          chi_squared_per_dof_dict = MakeDictionaryEntry(chi_squared_per_dof_dict, [f"chi_squared_per_dof_{sim_file_name}",uc_name,col], chi_squared_per_dof)

          total_uc_chi_squared_per_dof += chi_squared_per_dof
          count_uc_chi_squared_per_dof += 1

        # Add sum and mean to dictionaries
        if count_uc_chi_squared_per_dof > 0:
          if add_sum:
            chi_squared_per_dof_dict[f"chi_squared_per_dof_{sim_file_name}"][uc_name]["sum"] =  total_uc_chi_squared_per_dof
          if add_mean:
            chi_squared_per_dof_dict[f"chi_squared_per_dof_{sim_file_name}"][uc_name]["mean"] =  total_uc_chi_squared_per_dof/count_uc_chi_squared_per_dof
          
        total_sim_file_chi_squared_per_dof += total_uc_chi_squared_per_dof
        count_sim_file_chi_squared_per_dof += count_uc_chi_squared_per_dof

        # Add sum and mean to dictionaries
        if count_sim_file_chi_squared_per_dof > 0:
          if add_sum:
            chi_squared_per_dof_dict[f"chi_squared_per_dof_{sim_file_name}"]["sum"] =  total_sim_file_chi_squared_per_dof
          if add_mean:
            chi_squared_per_dof_dict[f"chi_squared_per_dof_{sim_file_name}"]["mean"] =  total_sim_file_chi_squared_per_dof/count_sim_file_chi_squared_per_dof

    return chi_squared_dict, dof_dict, chi_squared_per_dof_dict


  def GetKLDivergence(self, add_sum=True, add_mean=True):
      
    # Make histograms is they don't exist
    if not self.hists_exist:
      self.MakeHistograms()

    # Setup dictionaries
    kl_divergence_dict = {}

    # Loop through the sim files
    for sim_file_name, sim_hist_per_file in self.sim_hists.items():
      total_sim_file_kl_divergence = 0
      count_sim_file_kl_divergence = 0
      # Loop through the unique combinations
      for uc_name, sim_hist in sim_hist_per_file.items():
        total_uc_kl_divergence = 0
        count_uc_kl_divergence = 0
        # Loop through the columns
        for col in sim_hist.keys():

          # Calculate KL divergence
          kl_divergence = float(np.sum(self.synth_hists[sim_file_name][uc_name][col] * np.log(self.synth_hists[sim_file_name][uc_name][col] / self.sim_hists[sim_file_name][uc_name][col])))

          # Add to dictionaries
          kl_divergence_dict = MakeDictionaryEntry(kl_divergence_dict, [f"kl_divergence_{sim_file_name}",uc_name,col], kl_divergence)

          total_uc_kl_divergence += kl_divergence
          count_uc_kl_divergence += 1

        # Add sum and mean to dictionaries
        if count_uc_kl_divergence > 0:
          if add_sum:
            kl_divergence_dict[f"kl_divergence_{sim_file_name}"][uc_name]["sum"] =  total_uc_kl_divergence
          if add_mean:
            kl_divergence_dict[f"kl_divergence_{sim_file_name}"][uc_name]["mean"] =  total_uc_kl_divergence/count_uc_kl_divergence
          
        total_sim_file_kl_divergence += total_uc_kl_divergence
        count_sim_file_kl_divergence += count_uc_kl_divergence

        # Add sum and mean to dictionaries
        if count_sim_file_kl_divergence > 0:
          if add_sum:
            kl_divergence_dict[f"kl_divergence_{sim_file_name}"]["sum"] =  total_sim_file_kl_divergence
          if add_mean:
            kl_divergence_dict[f"kl_divergence_{sim_file_name}"]["mean"] =  total_sim_file_kl_divergence/count_sim_file_kl_divergence

    return kl_divergence_dict