import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml

import numpy as np
import pandas as pd

from functools import partial

from data_processor import DataProcessor
from plotting import plot_calibration_curve
from useful_functions import GetDefaultsInModel, InitiateDensityModel, LoadConfig, MakeDirectories, Translate

class Calibration():
  """
  A class to perform calibration of the predicted likelihood ratio from a density model.
  """

  def __init__(self):

    self.cfg = None
    self.open_cfg = None
    self.density_model = None
    self.model_input = "models/"
    self.data_input = None
    self.reference_data_input = None
    self.val_info = {}
    self.n_bins = 10
    self.ignore_quantile = 0.01
    self.data_output = "data/"
    self.plots_output = "plots/"
    self.sim_type = "val"
    self.extra_plot_name = ""
    self.verbose = True


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

    # Open config
    if self.verbose:
      print("- Loading in config")
    cfg = self.open_cfg if self.open_cfg is not None else LoadConfig(self.cfg)

    # Open parameters
    if self.verbose:
      print("- Loading in parameters")
    with open(self.density_model["parameters"], 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build the density model
    if self.verbose:
      print("- Building density network")
    density_model_name = f"{self.model_input}/{self.density_model['name']}/{parameters['file_name']}"
    with open(f"{density_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    network = InitiateDensityModel(
      architecture,
      self.density_model['file_loc'],
      options = {
        "data_parameters" : parameters["density"],
      }
    )
    network.Load(name=f"{density_model_name}.h5")

    # Build the reference (172.5 GeV) and test mass hypotheses
    if self.verbose:
      print("- Building reference and test mass hypotheses")
    reference_parameters = GetDefaultsInModel(parameters["file_name"], cfg)
    for lnN in parameters["yields"]["lnN"].keys():
      reference_parameters[lnN] = 0.0
    test_parameters = dict(reference_parameters)
    test_parameters.update(self.val_info)

    Y_columns = parameters["density"]["Y_columns"]
    Y_reference = pd.DataFrame({k: [v] for k, v in reference_parameters.items() if k in Y_columns})
    Y_test = pd.DataFrame({k: [v] for k, v in test_parameters.items() if k in Y_columns})

    # Load the MC truth events simulated under the test and reference hypotheses
    if self.verbose:
      print("- Loading in MC truth events")
    dp_test = DataProcessor(
      [self.data_input],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : parameters["density"],
      }
    )
    dp_reference = DataProcessor(
      [self.reference_data_input],
      "parquet",
      wt_name = "wt",
      options = {
        "parameters" : parameters["density"],
      }
    )

    def add_likelihood_ratio(df, network, Y_test, Y_reference):
      log_prob_test = network.Probability(df, Y_test, return_log_prob=True)
      log_prob_reference = network.Probability(df, Y_reference, return_log_prob=True)
      log_ratio = np.clip(log_prob_test - log_prob_reference, -700.0, 700.0)
      df["likelihood_ratio"] = np.exp(log_ratio)
      return df

    add_lr = partial(add_likelihood_ratio, network=network, Y_test=Y_test, Y_reference=Y_reference)

    # Bin the predicted likelihood ratio, using the reference (H0) sample to set the range
    if self.verbose:
      print(f"- Binning into {self.n_bins} likelihood ratio bins")
    bins = dp_reference.GetFull(
      method = "bins_with_equal_spacing",
      column = "likelihood_ratio",
      bins = self.n_bins,
      ignore_quantile = self.ignore_quantile,
      functions_to_apply = [add_lr],
    )

    # Get the binned event counts from each hypothesis
    if self.verbose:
      print("- Computing the binned event counts for each hypothesis")
    hist_test, hist_test_uncert, bin_edges = dp_test.GetFull(
      method = "histogram_and_uncert",
      column = "likelihood_ratio",
      bins = bins,
      functions_to_apply = [add_lr],
    )
    hist_reference, hist_reference_uncert, _ = dp_reference.GetFull(
      method = "histogram_and_uncert",
      column = "likelihood_ratio",
      bins = bins,
      functions_to_apply = [add_lr],
    )
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    # MC estimate of the true likelihood ratio per bin, from the ratio of the two hypotheses' binned counts
    non_zero = (hist_test > 0) & (hist_reference > 0)
    mc_ratio = np.divide(hist_test, hist_reference, out=np.full_like(bin_centers, np.nan), where=non_zero)
    mc_ratio_uncert = np.full_like(bin_centers, np.nan)
    mc_ratio_uncert[non_zero] = mc_ratio[non_zero] * np.sqrt(
      (hist_test_uncert[non_zero]/hist_test[non_zero])**2 + (hist_reference_uncert[non_zero]/hist_reference[non_zero])**2
    )

    # Write out results
    if self.verbose:
      print("- Writing calibration yaml")
    results = {
      "bin_edges" : [float(i) for i in bin_edges],
      "predicted_likelihood_ratio" : [float(i) for i in bin_centers],
      "mc_estimated_likelihood_ratio" : [None if np.isnan(i) else float(i) for i in mc_ratio],
      "mc_estimated_likelihood_ratio_uncert" : [None if np.isnan(i) else float(i) for i in mc_ratio_uncert],
      "val_info" : {k: float(v) for k, v in self.val_info.items()},
    }
    output_name = f"{self.data_output}/calibration{self.extra_plot_name}.yaml"
    MakeDirectories(output_name)
    with open(output_name, 'w') as yaml_file:
      yaml.dump(results, yaml_file, default_flow_style=False)

    # Make calibration plot
    if self.verbose:
      print("- Making calibration plot")
    test_text = ", ".join([f"{Translate(k)}={round(v,2)} GeV" for k, v in self.val_info.items()])
    reference_text = ", ".join([f"{Translate(k)}={round(v,2)} GeV" for k, v in reference_parameters.items() if k in self.val_info])
    axis_text = f"$H_{{1}}$: {test_text}\n$H_{{0}}$: {reference_text}"

    plot_calibration_curve(
      bin_centers,
      mc_ratio,
      mc_ratio_uncert,
      name = f"{self.plots_output}/calibration{self.extra_plot_name}",
      axis_text = axis_text,
    )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    outputs.append(f"{self.data_output}/calibration{self.extra_plot_name}.yaml")
    outputs.append(f"{self.plots_output}/calibration{self.extra_plot_name}.pdf")
    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    inputs += [self.cfg]
    inputs += [self.density_model["parameters"]]

    inputs += [f"{self.model_input}/{self.density_model['name']}/{self.density_model['file_name']}_architecture.yaml"]
    inputs += [f"{self.model_input}/{self.density_model['name']}/{self.density_model['file_name']}.h5"]
    inputs += list(self.data_input)
    inputs += list(self.reference_data_input)

    return inputs

