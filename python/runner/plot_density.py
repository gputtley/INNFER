import os
import yaml

from functools import partial

from data_processor import DataProcessor
from plotting import plot_histograms_with_ratio
from useful_functions import Translate, RoundToSF, LoadConfig, GetParametersInModel

class PlotDensity():

  def __init__(self):
    """
    A class to plot the comparison between train and test datasets, and synthetic samples.
    """
    self.cfg = None
    self.file_name = None
    self.parameters = None
    self.data_input = "data/"
    self.evaluate_input = "data/"
    self.plots_output = "plots/"
    self.verbose = True 
    self.n_plots = 20

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

    # Open parameters file
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Make the data processors
    for tt in ["train","test"]:
      if self.verbose:
        print(f"- Making plots for {tt} conditions")

      Y_wt_dataset = [f"{self.data_input}/{i}_{tt}.parquet" for i in (["Y","wt"] if len(parameters["density"]["Y_columns"]) > 0 else ["wt"])]
      X_dataset = f"{self.data_input}/X_{tt}.parquet"
      synth_dataset = f"{self.evaluate_input}/synth_{tt}.parquet"
      input_dp = DataProcessor(
        [Y_wt_dataset+[X_dataset]],
        "parquet",
        options = {
          "wt_name" : "wt",
          "parameters" : parameters["density"],
          "functions" : ["untransform"]
        }
      )
      synth_dp = DataProcessor(
        [Y_wt_dataset+[synth_dataset]],
        "parquet",
        options = {
          "wt_name" : "wt",
          "parameters" : parameters["density"],
          "functions" : ["untransform"]
        }
      )

      # Loop through conditions
      sels = {"Inclusive": None}
      names = {"Inclusive": "inclusive"}
      for condition in parameters["density"]["Y_columns"]:
        cond_bins = input_dp.GetFull(method="bins_with_equal_stats", column=condition, bins=self.n_plots, ignore_quantile=0.00)
        for i in range(len(cond_bins)-1):
          name = rf"{RoundToSF(cond_bins[i],2)} $\leq$ {Translate(condition)} < {RoundToSF(cond_bins[i+1],2)}"
          sels[name] = f"(({condition} >= {cond_bins[i]}) & ({condition} < {cond_bins[i+1]}))"
          names[name] = f"{condition}_bin_{i}"


      # Loop through columns
      for col in parameters["density"]["X_columns"] + parameters["density"]["Y_columns"]:
        for sel_name, sel in sels.items():

          # get nominal sums
          bins = input_dp.GetFull(method="bins_with_equal_spacing", column=col, bins=20, ignore_quantile=0.02, extra_sel=sel, ignore_discrete=True)
          input_hist, input_hist_uncerts, _ = input_dp.GetFull(method="histogram_and_uncert", column=col, bins=bins, extra_sel=sel)
          synth_hist, synth_hist_uncerts, _ = synth_dp.GetFull(method="histogram_and_uncert", column=col, bins=bins, extra_sel=sel)

          plot_histograms_with_ratio(
            [[synth_hist, input_hist]],
            [[synth_hist_uncerts, input_hist_uncerts]],
            [["Synthetic", "Simulated"]],
            bins,
            xlabel = col,
            ylabel="Events",
            name=f"{self.plots_output}/comparison_{col}_{tt}_{names[sel_name]}",      
            ratio_range = [0.9,1.1],
            first_ratio = True,
            axis_text=sel_name
          )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)
  
    # Add plots
    X_columns = cfg["variables"]
    Y_columns = GetParametersInModel(self.file_name, cfg, only_density=True)
    names = ["inclusive"]
    for col in Y_columns:
      names.extend([f"{col}_bin_{i}" for i in range(self.n_plots)])

    for tt in ["train","test"]:
      for col in X_columns + Y_columns:
        for name in names:
          outputs.append(f"{self.plots_output}/comparison_{col}_{tt}_{name}.pdf")

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    # Add parameters
    inputs.append(self.parameters)

    # Add data sets
    for tt in ["train","test"]:
      inputs.extend([f"{self.data_input}/{i}_{tt}.parquet" for i in ["Y","wt"]])
      inputs.append(f"{self.data_input}/X_{tt}.parquet")
      inputs.append(f"{self.evaluate_input}/synth_{tt}.parquet")

    return inputs

        