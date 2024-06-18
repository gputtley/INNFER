import copy
import yaml
import time
import numpy as np
from scipy.optimize import minimize
from useful_functions import GetYName, MakeDirectories

class Likelihood():
  """
  A class representing a likelihood function.
  """
  def __init__(self, models, likelihood_type="unbinned_extended", parameters={}, data_parameters={}):
    """
    Initializes a Likelihood object.

    Args:
        models (dict): A dictionary containing model names as keys and corresponding probability density functions (PDFs) as values.
        type (str): Type of likelihood, either "binned", "binned_extended", "unbinned" or "unbinned_extended" (default is "unbinned").
        parameters (dict): A dictionary containing model parameters (default is an empty dictionary).
        data_parameters (dict): A dictionary containing data parameters (default is an empty dictionary).
    """
    self.type = likelihood_type # unbinned, unbinned_extended, binned, binned_extended
    self.models = models
    self.parameters = parameters
    self.data_parameters = data_parameters
    self.Y_columns = self._MakeY()

    # saved parameters
    self.best_fit = None
    self.best_fit_nll = None

  def _MakeY(self):
    """
    Internal method to create a list of Y column names.

    Returns:
        list: A list of column names for the Y-values used in the likelihood calculation.

    """
    Y_columns = []
    if "rate_parameters" in self.parameters.keys():
      Y_columns += ["mu_"+rate_parameter for rate_parameter in self.parameters["rate_parameters"]]
    for _, data_params in self.data_parameters.items():
      Y_columns += [name for name in data_params["Y_columns"] if name not in Y_columns]
    return sorted(Y_columns)


  def Run(self, X_dps, Y, wts=None, return_ln=True, before_sum=False, multiply_by=1):
    """
    Evaluates the likelihood for given data.

    Args:
        X (array): The independent variables.
        Y (array): The model parameters.
        wts (array): The weights for the data points (optional).
        return_ln (bool): Whether to return the natural logarithm of the likelihood (optional).

    Returns:
        float: The likelihood value.

    """
    if self.type == "unbinned":
      lkld_val = self.Unbinned(X_dps, Y, wts=wts, return_ln=return_ln)
    elif self.type == "unbinned_extended":
      lkld_val = self.UnbinnedExtended(X_dps, Y, wts=wts, return_ln=return_ln)
    elif self.type == "binned":
      lkld_val = self.Binned(X_dps, Y, wts=wts, return_ln=return_ln)
    elif self.type == "binned_extended":
      lkld_val = self.BinnedExtended(X_dps, Y, wts=wts, return_ln=return_ln)

    return lkld_val*multiply_by

  def _LogSumExpTrick(self, ln_a, ln_b):
    mask = ln_b > ln_a
    ln_a[mask], ln_b[mask] = ln_b[mask], ln_a[mask]
    return ln_a + np.log1p(np.exp(ln_b - ln_a))

  def _GetYield(self, file_name, Y):
    yd = self.models["yields"][file_name](Y)
    return yd

  def _GetCombinedPDF(self, X, Y, wts=None, before_sum=False):

    first_loop = True
    sum_rate_params = 0.0

    # Loop through pdf models
    for name, pdf in self.models["pdfs"].items():

      # Get model scaling
      if "mu_"+name in self.Y_columns:
        rate_param = Y[self.Y_columns.index("mu_"+name)]
      else:
        rate_param = 1.0
      if rate_param == 0.0: continue
      rate_param *= self._GetYield(name, Y)
      sum_rate_params += rate_param

      # Get rate times probability
      log_p = pdf.Probability(X, Y, return_log_prob=True)
      ln_lklds_with_rate_params = np.log(rate_param) + log_p

      # Sum together probabilities of different files
      if first_loop:
        ln_lklds = copy.deepcopy(ln_lklds_with_rate_params)
        first_loop = False
      else:
        ln_lklds = self._LogSumExpTrick(ln_lklds,ln_lklds_with_rate_params)

    # Rescale so total pdf integrate to 1
    ln_lklds -= np.log(sum_rate_params)

    # Return probability before sum
    if before_sum:
      return ln_lklds

    # Weight the events
    if wts is not None:
      ln_lklds = wts.flatten()*ln_lklds

    # Product the events
    ln_lkld = np.sum(ln_lklds, dtype=np.float128)

    return ln_lkld

  def _GetConstraint(self, Y):

    # Add constraints
    if "nuisance_constraints" in self.parameters.keys():
      for k, v in self.parameters["nuisance_constraints"].items():
        if v == "Gaussian":
          constraint = self._Gaussian(Y[self.Y_columns.index(k)])
        elif v == "LogNormal":
          constraint = self._LogNormal(Y[self.Y_columns.index(k)])
        ln_lkld += np.log(constraint)


  def _CustomDPMethodForCombinedPDF(self, tmp, out, options={}):

    tmp_total = self._GetCombinedPDF(tmp, options["Y"])

    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      out += tmp_total

    return out

  def Unbinned(self, X_dps, Y, wts=None, return_ln=True, verbose=True):
    """
    Computes the likelihood for unbinned data.

    Args:
        X (array): The independent variables.
        Y (array): The model parameters.
        wts (array): The weights for the data points (optional).
        return_ln (bool): Whether to return the natural logarithm of the likelihood (optional).

    Returns:
        float: The likelihood value.

    """
    start_time = time.time()

    ln_lkld = 0
    for X_dp in X_dps:
      ln_lkld += X_dp.GetFull(
        method = "custom",
        functions_to_apply = ["untransform"],
        custom = self._CustomDPMethodForCombinedPDF,
        custom_options = {"Y" : Y}
      )

    end_time = time.time()

    if verbose:
      print(f">> Y={Y.to_numpy().flatten()}, lnL={ln_lkld}, time={round(end_time-start_time,2)}")

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)
