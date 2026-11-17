import pickle
import yaml

import numpy as np

from scipy.stats import norm

from data_processor import DataProcessor  
from plotting import plot_histograms
from useful_functions import LoadConfig, GetVariables, Translate

class PlotDensityTransform():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.cfg = None
    self.open_cfg = None
    self.parameters = None
    self.data_input = "data"
    self.plots_output = "plots"
    self.data_split = "train"
    self.category = None

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
    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      if self.verbose:
        print("- Loading in config")
      cfg = LoadConfig(self.cfg)

    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build data processor
    dp = DataProcessor(
      [[f"{self.data_input}/X_{self.data_split}.parquet", f"{self.data_input}/Y_{self.data_split}.parquet", f"{self.data_input}/wt_{self.data_split}.parquet"]], 
      "parquet",
      options = {
        "wt_name" : "wt",
        "selection" : None,
        "parameters" : parameters["density"]
      }
    )

    gaussian_transform = False
    if "density_pretransform_to_gaussian" in cfg["preprocess"]:
      if cfg["preprocess"]["density_pretransform_to_gaussian"]:
        gaussian_transform = True

    for col in GetVariables(cfg, category=self.category):

      quantiles = dp.GetFull(method="quantiles", quantiles=[0.001,0.999], column=col, functions_to_apply=["untransform"], single_batch=True)
      bins = np.linspace(quantiles[0], quantiles[1], 10000)

      if gaussian_transform:
        cdf_spline = f"{self.data_input}/X_spline_{col}.pkl"

        # open the spline file
        with open(cdf_spline, 'rb') as file:
          cdf_spline_func = pickle.load(file)
        y = cdf_spline_func(bins)

        plot_histograms(
          bins, 
          [y],
          ["CDF spline"],
          x_label = Translate(col),
          y_label = "CDF",
          name = f"{self.plots_output}/cdf_spline_{col}"
        )

        transform = norm.ppf(y)

        plot_histograms(
          bins, 
          [transform],
          ["Gaussian transform"],
          x_label = Translate(col),
          y_label = f"Spline Transformed {Translate(col)}",
          name = f"{self.plots_output}/gaussian_transform_{col}"
        )

        # plot gradient of spline
        y_grad = np.gradient(y, bins)

        plot_histograms(
          bins,
          [y_grad],
          ["Gradient of CDF spline"],
          x_label = Translate(col),
          y_label = "Gradient",
          name = f"{self.plots_output}/cdf_spline_gradient_{col}"
        )

        # plot gradient of Gaussian transform
        transform_grad = np.gradient(transform, bins)

        plot_histograms(
          bins,
          [transform_grad],
          ["Gradient of Gaussian transform"],
          x_label = Translate(col),
          y_label = "Gradient",
          name = f"{self.plots_output}/gaussian_transform_gradient_{col}"
        )

        # plot gradient but with x on 0 to 1
        quantile_range = quantiles[1] - quantiles[0]
        y_min, y_max = norm.ppf(cdf_spline_func(np.array([quantiles[0], quantiles[1]])))
        y_range = y_max - y_min
        x = np.linspace(y_min, y_max, 10000)
        bins = x * (quantile_range / y_range) + quantiles[0] - quantile_range * y_min / y_range
        cdf = cdf_spline_func(bins)
        y = norm.ppf(cdf)
        y_grad_ranged = np.gradient(y, x)

        plot_histograms(
          x,
          [y_grad_ranged],
          ["Gradient of Gaussian transform with consistent x range"],
          x_label = Translate(col),
          y_label = "Gradient",
          name = f"{self.plots_output}/gaussian_transform_gradient_ranged_{col}"
        )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """

    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      if self.verbose:
        print("- Loading in config")
      cfg = LoadConfig(self.cfg)

    outputs = []

    for col in cfg["variables"]:
      outputs += [
        f"{self.plots_output}/gaussian_transform_{col}.pdf",
        f"{self.plots_output}/cdf_spline_{col}.pdf",
        f"{self.plots_output}/gaussian_transform_gradient_{col}.pdf",
        f"{self.plots_output}/gaussian_transform_gradient_ranged_{col}.pdf"
      ]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """

    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      if self.verbose:
        print("- Loading in config")
      cfg = LoadConfig(self.cfg)

    inputs = []
    inputs += [self.parameters]
    inputs += [f"{self.data_input}/X_{self.data_split}.parquet", f"{self.data_input}/Y_{self.data_split}.parquet", f"{self.data_input}/wt_{self.data_split}.parquet"]

    gaussian_transform = False
    if "density_pretransform_to_gaussian" in cfg["preprocess"]:
      if cfg["preprocess"]["density_pretransform_to_gaussian"]:
        gaussian_transform = True

    if gaussian_transform:
      for col in cfg["variables"]:
        inputs += [f"{self.data_input}/X_spline_{col}.pkl"]

    return inputs

        