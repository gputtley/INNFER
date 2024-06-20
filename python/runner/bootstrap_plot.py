import yaml
import numpy as np

from useful_functions import MakeDirectories, GetYName
from plotting import plot_histograms

class BootstrapPlot():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.column = None

    self.data_input = "data"
    self.plots_output = "plots"    
    self.extra_file_name = ""
    self.bins = 20
    self.extra_plot_name = ""

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.extra_file_name != "":
      self.extra_file_name = f"_{self.extra_file_name}"

    if self.extra_plot_name != "":
      self.extra_plot_name = f"_{self.extra_plot_name}"

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    # Open bootstraps
    with open(f"{self.data_input}/bootstrap_results_{self.column}{self.extra_file_name}.yaml", 'r') as yaml_file:
      bootstrap_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

    hist, bins = np.histogram(bootstrap_results_info["results"], bins=self.bins, density=True)
    hist_total, _ = np.histogram(bootstrap_results_info["results"], bins=bins)
    hist_total_err = np.sqrt(hist_total)
    bin_widths = np.diff(bins)
    counts_per_bin = np.sum(hist_total * bin_widths)
    hist_err = hist_total_err/counts_per_bin

    mean = bootstrap_results_info["mean"]
    std = bootstrap_results_info["std"]

    plot_extra_name = GetYName(bootstrap_results_info["row"], purpose="plot", prefix="y=")

    def gauss(x):
      exponent = -0.5 * ((x - mean) / std) ** 2    
      coefficient = 1 / (np.sqrt(2 * np.pi) * std)    
      return coefficient * np.exp(exponent)

    plot_histograms(
      bins[:-1],
      [],
      [],
      error_bar_hists = [hist],
      error_bar_hist_errs = [hist_err],
      error_bar_names = [f"{len(bootstrap_results_info['results'])} Bootstraps"],
      title_right = plot_extra_name,
      name = f"{self.plots_output}/bootstrap_distribution_{self.column}{self.extra_file_name}{self.extra_plot_name}",
      x_label = f"Best Fit {self.column}",
      y_label = "Density of Bootstraps",
      drawstyle = "steps",
      smooth_func = gauss,
      smooth_func_name = rf"Gauss($\mu$={round(mean,2)},$\sigma$={round(std,2)})",
      smooth_func_color = "green",
      vertical_lines = [bootstrap_results_info["row"][bootstrap_results_info["columns"].index(self.column)], mean, mean-std, mean+std],
      vertical_line_names = ["Truth", "Mean", r"$\pm 1 \sigma$", None],
      vertical_line_colors = ["blue", "red", "orange", "orange"]
    )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [
      f"{self.plots_output}/bootstrap_distribution_{self.column}{self.extra_file_name}{self.extra_plot_name}.pdf"
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [
      f"{self.data_input}/bootstrap_results_{self.column}{self.extra_file_name}.yaml"
    ]
    return inputs
