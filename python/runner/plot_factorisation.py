import numpy as np

from plotting import plot_correlation_matrix
from useful_functions import GetDictionaryEntryFromYaml

class PlotFactorisation():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.metric = ["chi_squared_per_dof","mean"]
    self.metrics_files = None
    self.plots_output = None
    self.extra_plot_name = ""
    self.verbose = False
    self.columns = None
    self.x_shift = None
    self.y_shift = None

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
        
    # Load the shifted metrics
    metric_values = []
    for ind_1 in range(len(self.metrics_files)):
      metric_values.append([])
      for ind_2 in range(len(self.metrics_files[ind_1])):
        if self.metrics_files[ind_1][ind_2] is not None:
          metric_values[-1].append(GetDictionaryEntryFromYaml(self.metrics_files[ind_1][ind_2], self.metric))
        else:
          metric_values[-1].append(np.nan)

    plot_name = f"{self.plots_output}/FactorisationMatrix_{'_'.join(self.metric)}"
    if self.extra_plot_name != "":
      plot_name += f"_{self.extra_plot_name}"

    # Plot a correlation matrix
    plot_correlation_matrix(
      np.array(metric_values),
      self.columns,
      output_name = plot_name,
      xlabel = f"Parameter shifted {self.x_shift}",
      ylabel = f"Parameter shifted {self.y_shift}",
      zlabel = f", ".join(self.metric),
    )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    plot_name = f"{self.plots_output}/FactorisationMatrix_{'_'.join(self.metric)}"
    if self.extra_plot_name != "":
      plot_name += f"_{self.extra_plot_name}"
    outputs += [f"{plot_name}.pdf"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []
    inputs += [i for i in np.array(self.metrics_files).flatten() if i is not None]

    return inputs

        