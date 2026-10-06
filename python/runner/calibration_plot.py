import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml

import numpy as np

from functools import partial

from data_processor import DataProcessor
from plotting import plot_calibration_curve
from infer import Infer
from useful_functions import MakeDirectories, Translate

class CalibrationPlot(Infer):
  """
  A class to calibrate the event likelihood ratio for a process mixture.
  """

  def __init__(self):

    super().__init__()

    self.cfg = None
    self.open_cfg = None
    self.file_name = None
    self.category = None
    self.reference_val_ind = None
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
    self.extra_density_model_name = ""
    self.likelihood_type = "unbinned"


  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.likelihood_type != "unbinned":
      raise ValueError("CalibrationPlot requires likelihood_type='unbinned'")

    if self.classifier_divide_by_nominal:
      self.classifier_models = {
        category: {
          process: [model for model in models if any(
            model["parameter"] in Y.columns and float(Y.iloc[0][model["parameter"]]) != 0.0
            for Y in [self.true_Y, self.initial_best_fit_guess]
          )]
          for process, models in processes.items()
        }
        for category, processes in self.classifier_models.items()
      }

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"


  def Run(self):

    # Build the same model components and yield functions used by inference
    if self.verbose:
      print("- Building likelihood models")
    if self.yields is None:
      # A common effective-event scaling cancels in the event ratio. Keep the
      # physical yields here so both hypothesis samples use the same units.
      scale_to_eff_events = self.scale_to_eff_events
      self.scale_to_eff_events = False
      try:
        self.yields = self._BuildYieldFunctions()
      finally:
        self.scale_to_eff_events = scale_to_eff_events
    if self.lkld is None:
      self.lkld = self._BuildLikelihood()

    # Build the reference and test hypotheses, including ratio/yield parameters
    if self.verbose:
      print("- Building reference and test hypotheses")
    Y_reference = self.initial_best_fit_guess.copy(deep=True)
    Y_test = self.true_Y.copy(deep=True)
    reference_parameters = Y_reference.iloc[0].to_dict()

    # Stack processes vertically; each process has aligned feature/weight files
    if self.verbose:
      print("- Loading in simulation events")
    dp_test = DataProcessor(
      list(self.data_input[self.category].values()),
      "parquet",
      wt_name = "wt",
      options = {"hold_dataset_in_memory" : self.hold_dataset_in_memory},
    )
    dp_reference = DataProcessor(
      list(self.reference_data_input[self.category].values()),
      "parquet",
      wt_name = "wt",
      options = {"hold_dataset_in_memory" : self.hold_dataset_in_memory},
    )

    add_lr = partial(self._AddLikelihoodRatio, Y_test=Y_test, Y_reference=Y_reference)

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

    # Shape likelihoods compare probability densities rather than event yields
    total_test = float(dp_test.GetFull(method="sum"))
    total_reference = float(dp_reference.GetFull(method="sum"))
    if total_test <= 0 or total_reference <= 0:
      raise ValueError("CalibrationPlot requires positive total weights for both hypotheses")
    hist_test /= total_test
    hist_test_uncert /= total_test
    hist_reference /= total_reference
    hist_reference_uncert /= total_reference

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
      "reference_parameters" : {k: float(v) for k, v in reference_parameters.items()},
      "test_parameters" : {k: float(v) for k, v in Y_test.iloc[0].to_dict().items()},
      "file_name" : self.file_name,
      "category" : self.category,
      "val_ind" : self.val_ind,
      "reference_val_ind" : self.reference_val_ind,
      "likelihood_type" : self.likelihood_type,
    }
    output_name = f"{self.data_output}/calibration_plot{self.extra_plot_name}.yaml"
    MakeDirectories(output_name)
    with open(output_name, 'w') as yaml_file:
      yaml.dump(results, yaml_file, default_flow_style=False)

    # Make calibration plot
    if self.verbose:
      print("- Making calibration plot")
    test_text = ", ".join([f"{Translate(k, only_val=True)}={round(v,2)}{Translate(k, only_unit=True)}" for k, v in self.val_info.items()])
    reference_text = ", ".join([f"{Translate(k, only_val=True)}={round(v,2)}{Translate(k, only_unit=True)}" for k, v in reference_parameters.items() if k in self.val_info])
    axis_text = f"$H_{{1}}$: {test_text}\n$H_{{0}}$: {reference_text}"

    plot_calibration_curve(
      bin_centers,
      mc_ratio,
      mc_ratio_uncert,
      name = f"{self.plots_output}/calibration_plot{self.extra_plot_name}",
      axis_text = axis_text,
    )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    outputs.append(f"{self.data_output}/calibration_plot{self.extra_plot_name}.yaml")
    outputs.append(f"{self.plots_output}/calibration_plot{self.extra_plot_name}.pdf")
    return outputs


  def _AddLikelihoodRatio(self, df, Y_test, Y_reference):
    """
    Add the per-event mixture likelihood ratio, including learned shape shifts.
    """
    log_values = []
    for Y in [Y_test, Y_reference]:
      log_probs = self.lkld._GetLogProbs(df, Y, gradient=[0], category=self.category)
      log_value = self.lkld._GetH1({k: v[0] for k, v in log_probs.items()}, Y, log=True, category=self.category)
      log_value -= self.lkld._GetH2(Y, log=True, category=self.category)
      log_values.append(np.asarray(log_value).reshape(-1))
    log_ratio = log_values[0] - log_values[1]
    if not np.all(np.isfinite(log_ratio)):
      raise ValueError("Non-finite event likelihood ratio in calibration")
    df["likelihood_ratio"] = np.exp(np.clip(log_ratio, -700.0, 700.0))
    return df


  def Inputs(self):
    """
    Return all model, metadata, and hypothesis dataset dependencies.
    """
    inputs = [self.cfg] + super().Inputs()
    for files in self.reference_data_input[self.category].values():
      inputs += files
    return list(dict.fromkeys(inputs))
