import yaml

import numpy as np

from histogram_metrics import HistogramMetrics
from make_asimov import MakeAsimov
from multidim_metrics import MultiDimMetrics

from useful_functions import LoadConfig, MakeDirectories

class ValidationPerformanceMetrics():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.cfg = None
    self.sim_files = None
    self.synth_files = None
    self.sim_wt_name = "wt"
    self.synth_wt_name = "wt"
    self.data_output = None
    self.verbose = False

    self.do_histogrmam_metrics = True
    self.do_chi_squared = True
    self.do_kl_divergence = True

    self.do_multidimensional_dataset_metrics = True
    self.do_bdt_separation = True
    self.do_wasserstein = True
    self.do_sliced_wasserstein = True
    self.do_kmeans_chi_squared = False

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
    # Load config       
    if self.verbose:
      print(f"- Loading config")
    cfg = LoadConfig(self.cfg)

    # Initiate metrics
    self.metrics = {}

    # Histogram metrics
    if self.do_histogrmam_metrics:

      if self.verbose:
        print(f"- Running histogram metrics")

      hm = HistogramMetrics(
        self.sim_files, 
        self.synth_files,
        cfg["variables"],
        sim_wt_name = self.sim_wt_name,
        synth_wt_name = self.synth_wt_name,
      )

      # Get chi squared
      if self.do_chi_squared:
        chi_squared, dof_for_chi_squared, chi_squared_per_dof = hm.GetChiSquared()
        if len(chi_squared.keys()) != 0:
          self.metrics[f"chi_squared"] = chi_squared
          self.metrics[f"dof_for_chi_squared"] = dof_for_chi_squared
          self.metrics[f"chi_squared_per_dof"] = chi_squared_per_dof    

      # Get kl divergence
      if self.do_kl_divergence:
        kl_divergence = hm.GetKLDivergence()
        if len(kl_divergence.keys()) != 0:
          self.metrics[f"kl_divergence"] = kl_divergence

    # Multidimensional dataset metrics
    if self.do_multidimensional_dataset_metrics:

      if self.verbose:
        print(f"- Running multidimensional dataset metrics")

      mm = MultiDimMetrics(
        self.sim_files, 
        self.synth_files,
        cfg["variables"],
        sim_wt_name = self.sim_wt_name,
        synth_wt_name = self.synth_wt_name,
      )

      # Get BDT separation metric
      if self.do_bdt_separation:
        if self.verbose:
          print(" - Adding BDT separation")
        mm.AddBDTSeparation()
      
      # Get Wasserstein metric
      if self.do_wasserstein:
        if self.verbose:
          print(" - Adding Wasserstein")
        mm.AddWassersteinUnbinned()

      # Get sliced Wasserstein metric
      if self.do_sliced_wasserstein:
        if self.verbose:
          print(" - Adding sliced Wasserstein")
        mm.AddWassersteinSliced()

      # Get kmeans chi squared metric
      if self.do_kmeans_chi_squared:
        if self.verbose:
          print(" - Adding kmeans chi squared")
        mm.AddKMeansChiSquared()


      # Run metrics
      multidim_metrics = mm.Run()
      self.metrics = {**self.metrics, **multidim_metrics}

    # Write metrics to file
    out_name = f"{self.data_output}/metrics.yaml"
    MakeDirectories(out_name)
    with open(out_name, "w") as f:
      yaml.dump(self.metrics, f)


  def Inputs(self):
    """
    Return the inputs required for the code to run.
    """
    inputs = []
    inputs += list(np.array(self.sim_files).flatten())
    inputs += list(np.array(self.synth_files).flatten())

    return inputs
  
  def Outputs(self):
    """
    Return the outputs produced by the code.
    """
    outputs = []
    outputs += [f"{self.data_output}/metrics.yaml"]

    return outputs