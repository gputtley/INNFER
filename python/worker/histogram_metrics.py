import numpy as np

from data_processor import DataProcessor
from useful_functions import MakeDictionaryEntry

class HistogramMetrics():

  def __init__(self, sim_files, synth_files, columns):
    self.sim_files = sim_files
    self.synth_files = synth_files
    self.columns = columns

    self.hists_exist = False
    self.sim_hists = None
    self.sim_hist_uncerts = None
    self.synth_hists = None
    self.synth_hist_uncerts = None

  def MakeHistograms(self, n_bins=40):

    # Make DataProcessors
    sim_dp = DataProcessor(
      self.sim_files,
      "parquet",
      wt_name = "wt",
      options = {}
    )
    synth_dp = DataProcessor(
      self.synth_files,
      "parquet",
      wt_name = "wt",
      options = {}
    )

    # loop through columns
    self.sim_hists = {}
    self.sim_hist_uncerts = {}
    self.synth_hists = {}
    self.synth_hist_uncerts = {}
    self.bin_centers = {}

    for col in self.columns:

      if np.sum(sim_dp.num_batches) == 0:
        continue
      if np.sum(synth_dp.num_batches) == 0:
        continue
      if sim_dp.GetFull(method="count") == 0:
        continue
      if synth_dp.GetFull(method="count") == 0:
        continue

      # Make synth histograms
      synth_hist, synth_hist_uncert, bins = synth_dp.GetFull(
        method = "histogram_and_uncert",
        bins = n_bins,
        column = col,
        density = True,
        )

      # Make sim histograms            
      sim_hist, sim_hist_uncert, bins = sim_dp.GetFull(
        method = "histogram_and_uncert",
        bins = bins,
        column = col,
        density = True,
      )

      # Add to stores
      self.sim_hists[col] = sim_hist
      self.sim_hist_uncerts[col] = sim_hist_uncert
      self.synth_hists[col] = synth_hist
      self.synth_hist_uncerts[col] = synth_hist_uncert
      self.bin_centers[col] = (bins[:-1] + bins[1:]) / 2

    self.RemoveZeroBins()
    self.hists_exist = True


  def RemoveZeroBins(self):

    for col in self.sim_hists.keys():
      non_zero_sim_indices = np.where((self.sim_hists[col] > 0) & (self.synth_hists[col] > 0))[0]
      self.sim_hists[col] = self.sim_hists[col][non_zero_sim_indices]
      self.sim_hist_uncerts[col] = self.sim_hist_uncerts[col][non_zero_sim_indices]
      self.synth_hists[col] = self.synth_hists[col][non_zero_sim_indices]
      self.synth_hist_uncerts[col] = self.synth_hist_uncerts[col][non_zero_sim_indices]
      self.bin_centers[col] = self.bin_centers[col][non_zero_sim_indices]


  def GetChiSquared(self, add_sum=True, add_mean=True):

    # Make histograms is they don't exist
    if not self.hists_exist:
      self.MakeHistograms()

    # Setup dictionaries
    chi_squared_dict = {}
    dof_dict = {}
    chi_squared_per_dof_dict = {}

    # Loop through the columns
    total_chi_squared_per_dof = 0
    count_chi_squared_per_dof = 0
    for col in self.sim_hists.keys():  

      # Calculate chi squared
      chi_squared = float(np.sum((self.synth_hists[col]-self.sim_hists[col])**2/(self.synth_hist_uncerts[col]**2 + self.sim_hist_uncerts[col]**2)))
      dof = len(self.synth_hists[col])
      chi_squared_per_dof = chi_squared / dof

      # Add to dictionaries
      chi_squared_dict[col] = chi_squared
      dof_dict[col] = dof
      chi_squared_per_dof_dict[col] = chi_squared_per_dof
      total_chi_squared_per_dof += chi_squared_per_dof
      count_chi_squared_per_dof += 1

    # Add sum and mean to dictionaries
    if count_chi_squared_per_dof > 0:
      if add_sum:
        chi_squared_per_dof_dict["sum"] =  total_chi_squared_per_dof
      if add_mean:
        chi_squared_per_dof_dict["mean"] =  total_chi_squared_per_dof/count_chi_squared_per_dof

    return chi_squared_dict, dof_dict, chi_squared_per_dof_dict


  def GetKLDivergence(self, add_sum=True, add_mean=True):
      
    # Make histograms is they don't exist
    if not self.hists_exist:
      self.MakeHistograms()

    # Setup dictionaries
    kl_divergence_dict = {}

    # Loop through columns
    total_kl_divergence = 0
    count_kl_divergence = 0
    for col in self.sim_hists.keys():
      # Calculate KL divergence
      kl_divergence = float(np.sum(self.synth_hists[col] * np.log(self.synth_hists[col] / self.sim_hists[col])))

      # Add to dictionaries
      kl_divergence_dict = MakeDictionaryEntry(kl_divergence_dict, [f"kl_divergence",col], kl_divergence)

      total_kl_divergence += kl_divergence
      count_kl_divergence += 1
    # Add sum and mean to dictionaries
    if count_kl_divergence > 0:
      if add_sum:
        kl_divergence_dict["sum"] =  total_kl_divergence
      if add_mean:
        kl_divergence_dict["mean"] =  total_kl_divergence/count_kl_divergence

    return kl_divergence_dict