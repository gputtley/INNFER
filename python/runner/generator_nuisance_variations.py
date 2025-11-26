
import os

from data_processor import DataProcessor
from useful_functions import Translate, LoadConfig
from plotting import plot_learned_nuisance_variations

class GeneratorNuisanceVariations():

  def __init__(self):

    self.cfg = None
    self.nominal_sim_input = None
    self.up_sim_input = None
    self.down_sim_input = None
    self.nominal_asimov_input = None
    self.up_asimov_input = None
    self.down_asimov_input = None
    self.plots_output = None
    self.nuisance = None
    self.extra_plot_name = None
    self.file_name = None
    self.category = None
    self.verbose = True
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    self.n_bins=30


  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"


  def Run(self):

    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg)

    # Build data processors
    nominal_sim_dp = DataProcessor(
      [self.nominal_sim_input],
      "parquet",
      batch_size=self.batch_size,
      wt_name="wt",
    )
    up_sim_dp = DataProcessor(
      self.up_sim_input,
      "parquet",
      batch_size=self.batch_size,
      wt_name="wt",
    )
    down_sim_dp = DataProcessor(
      self.down_sim_input,
      "parquet",
      batch_size=self.batch_size,
      wt_name="wt",
    )
    nominal_asimov_dp = DataProcessor(
      [self.nominal_asimov_input],
      "parquet",
      batch_size=self.batch_size,
      wt_name="wt",
    )
    up_asimov_dp = DataProcessor(
      self.up_asimov_input,
      "parquet",
      batch_size=self.batch_size,
      wt_name="wt",
    )
    down_asimov_dp = DataProcessor(
      self.down_asimov_input,
      "parquet",
      batch_size=self.batch_size,
      wt_name="wt",
    )

    # Make histograms 
    for col in cfg["variables"]:
      
      nominal_sim_hist, nominal_sim_uncert, bins = nominal_sim_dp.GetFull(method="histogram_and_uncert", column=col, bins=self.n_bins)
      up_sim_hist, up_sim_uncert, _ = up_sim_dp.GetFull(method="histogram_and_uncert", column=col, bins=bins)
      down_sim_hist, down_sim_uncert, _ = down_sim_dp.GetFull(method="histogram_and_uncert", column=col, bins=bins)
      nominal_asimov_hist, nominal_asimov_uncert, _ = nominal_asimov_dp.GetFull(method="histogram_and_uncert", column=col, bins=bins)
      up_asimov_hist, up_asimov_uncert, _ = up_asimov_dp.GetFull(method="histogram_and_uncert", column=col, bins=bins)
      down_asimov_hist, down_asimov_uncert, _ = down_asimov_dp.GetFull(method="histogram_and_uncert", column=col, bins=bins)

      extra_name = ""
      if self.extra_plot_name != "":
        extra_name = f"_{self.extra_plot_name}"

      plot_learned_nuisance_variations(
        nominal_asimov_hist,
        up_asimov_hist,
        down_asimov_hist,
        nominal_sim_hist,
        up_sim_hist,
        down_sim_hist,
        nominal_sim_uncert,
        up_sim_uncert,
        down_sim_uncert,
        bins,
        xlabel=Translate(col),
        output_name=f"{self.plots_output}/nuisance_distribution_{self.nuisance}_{col}{extra_name}",
        line_caption = "Synthetic",
        errorbar_caption = "Simulated",
        axis_text = f"{Translate(self.file_name)} {Translate(self.category)} {Translate(self.nuisance)}"
      )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Loop through columns
    for col in cfg["variables"]:
      pass

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initiate inputs
    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add file inputs
    inputs += self.nominal_sim_input
    inputs += self.up_sim_input
    inputs += self.down_sim_input
    inputs += self.nominal_asimov_input
    inputs += self.up_asimov_input
    inputs += self.down_asimov_input

    return inputs

