import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import yaml
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize
from functools import partial
from pprint import pprint

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
    self.seed = 42
    self.verbose = True

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

  def Run(self, X_dps, Y, return_ln=True, multiply_by=1, Y_columns=[]):
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
    start_time = time.time()

    # Check type of Y
    if not isinstance(Y, pd.DataFrame):
      Y = pd.DataFrame([Y], columns=self.Y_columns, dtype=np.float64)

    # Run likelihood
    if self.type == "unbinned":
      lkld_val = self.Unbinned(X_dps, Y, return_ln=return_ln)
    elif self.type == "unbinned_extended":
      lkld_val = self.UnbinnedExtended(X_dps, Y, return_ln=return_ln)
    elif self.type == "binned":
      lkld_val = self.Binned(X_dps, Y, return_ln=return_ln)
    elif self.type == "binned_extended":
      lkld_val = self.BinnedExtended(X_dps, Y, return_ln=return_ln)

    end_time = time.time()
    if self.verbose:
      print(f"Y={Y.to_numpy().flatten()}, lnL={lkld_val}, time={round(end_time-start_time,2)}")

    return lkld_val*multiply_by

  def _LogSumExpTrick(self, ln_a, ln_b):
    mask = ln_b > ln_a
    ln_a[mask], ln_b[mask] = ln_b[mask], ln_a[mask]
    return ln_a + np.log1p(np.exp(ln_b - ln_a))

  def _GetYield(self, file_name, Y):
    yd = self.models["yields"][file_name](Y)
    return yd

  def _GetCombinedPDF(self, X, Y, wt_name=None, before_sum=False, normalise=True):

    first_loop = True
    sum_rate_params = 0.0

    # Loop through pdf models
    for name, pdf in self.models["pdfs"].items():

      # Get model scaling
      rate_param = self._GetYield(name, Y)
      if rate_param == 0: continue
      sum_rate_params += rate_param

      # Get rate times probability
      log_p = pdf.Probability(X.loc[:, self.data_parameters[name]["X_columns"]], Y, return_log_prob=True)
      #log_p = np.ones(len(X)).reshape(-1,1) # uncomment this for yield only unbinned_extended likelihood
      ln_lklds_with_rate_params = np.log(rate_param) + log_p

      # Sum together probabilities of different files
      if first_loop:
        ln_lklds = copy.deepcopy(ln_lklds_with_rate_params)
        first_loop = False
      else:
        ln_lklds = self._LogSumExpTrick(ln_lklds,ln_lklds_with_rate_params)

    if normalise:
      # Rescale so total pdf integrate to 1
      ln_lklds -= np.log(sum_rate_params)

    # Return probability before sum
    if before_sum:
      return ln_lklds

    # Weight the events
    if wt_name is not None:
      ln_lklds = X.loc[:,[wt_name]].to_numpy(dtype=np.float64)*ln_lklds

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

    tmp_total = self._GetCombinedPDF(tmp, options["Y"], wt_name=options["wt_name"], normalise=options["normalise"])

    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      out += tmp_total

    return out

  def Unbinned(self, X_dps, Y, return_ln=True, normalise=True):
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

    ln_lkld = 0
    tf.random.set_seed(self.seed)
    tf.keras.utils.set_random_seed(self.seed)
    for dps_ind, X_dp in enumerate(X_dps):
      dps_ln_lkld = X_dp.GetFull(
        method = "custom",
        #functions_to_apply = ["untransform"],
        custom = self._CustomDPMethodForCombinedPDF,
        custom_options = {"Y" : Y, "wt_name" : X_dp.wt_name, "normalise" : normalise}
      )
      ln_lkld += dps_ln_lkld

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

  def UnbinnedExtended(self, X_dps, Y, return_ln=True):

    # Get densities
    ln_lkld = self.Unbinned(X_dps, Y, return_ln=True, normalise=False)

    # Add poisson term
    total_rate = 0.0
    for name, pdf in self.models["yields"].items():
      total_rate += self._GetYield(name, Y)

    ln_lkld -= total_rate

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)


  def GetBestFit(self, X_dps, initial_guess, method="nominal", freeze={}, initial_step_size=0.05):
    """
    Finds the best-fit parameters using numerical optimization.

    Args:
        X (array): The independent variables.
        initial_guess (array): Initial values for the model parameters.
        wts (array): The weights for the data points (optional).

    """
    if method == "nominal":

      func_to_minimise = lambda Y: self.Run(X_dps, Y, multiply_by=-2, Y_columns=list(initial_guess.columns))
      result = self.Minimise(func_to_minimise, initial_guess.to_numpy().flatten(), freeze=freeze, initial_step_size=initial_step_size)

    elif method == "low_stat_high_stat":

      def NLL(Y): 
        nll = -2*self.Run(X_dps, Y, return_ln=True)
        return nll
      
      stats = 1000
      original_stats = len(X)
      rng = np.random.RandomState(seed=42)
      random_indices = rng.choice(original_stats, stats, replace=False)
      lowstat_X = X[random_indices,:]
      if wts is not None:
        old_sum_wts = np.sum(wts, dtype=np.float128)
        lowstat_wts = wts[random_indices,:]
        new_sum_wt = np.sum(lowstat_wts, dtype=np.float128)
        lowstat_wts *= old_sum_wts/new_sum_wt
      else:
        lowstat_wts = (original_stats/stats)*np.ones(original_stats)

      def NLL_lowstat(Y):
        nll = -2*self.Run(lowstat_X, Y, wts=lowstat_wts, return_ln=True)
        return nll
      
      result = self.Minimise(NLL, initial_guess, method="low_stat_high_stat", func_low_stat=NLL_lowstat, freeze=freeze)

    return result

  def GetScanXValues(self, X_dps, column, estimated_sigmas_shown=3.2, estimated_sigma_step=0.4, initial_step_fraction=0.001, min_step=0.1):
    """
    Computes the scan values for a given column.

    Args:
        X (array): The independent variables.
        column (str): The column name for the parameter to be scanned.
        wts (array): The weights for the data points (optional).
        estimated_sigmas_shown (int): Number of estimated sigmas to be shown in the scan (default is 5).
        estimated_sigma_step (float): Step size for estimated sigmas in the scan (default is 0.2).
        initial_step_fraction (float): Fraction of the initial value to use as the step size (default is 0.001).
        min_step (float): Minimum step size (default is 0.1).

    Returns:
        list: The scan values.
    """
    col_index = self.Y_columns.index(column)
    nll_could_be_neg = True
    while nll_could_be_neg:
      m1p1_vals = {}
      fin = True
      for sign in [1,-1]:
        Y = copy.deepcopy(self.best_fit)
        step = max(self.best_fit[col_index]*initial_step_fraction,min_step)
        Y[col_index] = self.best_fit[col_index] + sign*step
        nll = self.Run(X_dps, Y, return_ln=True, multiply_by=-2) - self.best_fit_nll
        if nll < 0.0:
          self.best_fit[col_index] = Y[col_index]
          self.best_fit_nll += nll
          fin = False
          break
        est_sig = np.sqrt(nll)
        m1p1_vals[sign] = step/est_sig
      if fin: nll_could_be_neg = False

    if self.verbose:
      print(f"Estimated result: {float(self.best_fit[col_index])} + {m1p1_vals[1]} - {m1p1_vals[-1]}")

    lower_scan_vals = [float(self.best_fit[col_index] - estimated_sigma_step*ind*m1p1_vals[-1]) for ind in range(int(np.ceil(estimated_sigmas_shown/estimated_sigma_step)),0,-1)]
    upper_scan_vals = [float(self.best_fit[col_index] + estimated_sigma_step*ind*m1p1_vals[1]) for ind in range(1,int(np.ceil(estimated_sigmas_shown/estimated_sigma_step))+1)]

    return lower_scan_vals + [float(self.best_fit[col_index])] + upper_scan_vals

  def Minimise(self, func, initial_guess, method="scipy", func_low_stat=None, freeze={}, initial_step_size=0.02):
    """
    Minimizes the given function using numerical optimization.

    Args:
        func (callable): The objective function to be minimized.
        initial_guess (array): Initial values for the parameters.

    Returns:
        tuple: Best-fit parameters and the corresponding function value.

    """

    # freeze
    if len(list(freeze.keys())) > 0:
      initial_guess_before = copy.deepcopy(initial_guess)
      initial_guess = [initial_guess_before[ind] for ind, col in enumerate(self.Y_columns) if col not in freeze.keys()]
      old_func = func

      def func(x):
        extended_x = []
        x_ind = 0
        for col in self.Y_columns:
          if col in freeze.keys():
            extended_x.append(freeze[col])
          else:
            extended_x.append(x[x_ind])
            x_ind += 1
        return old_func(extended_x)

    # get initial simplex
    n = len(initial_guess)
    initial_simplex = [initial_guess]
    for i in range(n):
      row = copy.deepcopy(initial_guess)
      row[i] += initial_step_size * max(row[i], 1.0)
      initial_simplex.append(row)

    # minimisation
    if method == "scipy":
      minimisation = minimize(func, initial_guess, method='Nelder-Mead', tol=0.001, options={'xatol': 0.001, 'fatol': 0.01, 'initial_simplex': initial_simplex})
      return minimisation.x, minimisation.fun
    
    elif method == "low_stat_high_stat":

      if self.verbose:
        print("Doing low stat minimisation first.")

      low_stat_minimisation = minimize(func_low_stat, initial_guess, method='Nelder-Mead', tol=1.0, options={'xatol': 0.001, 'fatol': 1.0, 'initial_simplex': initial_simplex})

      if self.verbose:
        print("Doing high stat minimisation.")

      # re get initial simplex
      n = len(low_stat_minimisation.x)
      initial_simplex = [low_stat_minimisation.x]
      for i in range(n):
        row = copy.deepcopy(low_stat_minimisation.x)
        row[i] += 0.1* initial_step_size * max(row[i], 1.0) # 10% of initial simplex
        initial_simplex.append(row)

      minimisation = minimize(func, low_stat_minimisation.x, method='Nelder-Mead', tol=0.1, options={'xatol': 0.001, 'fatol': 0.1, 'initial_simplex': initial_simplex})
      return minimisation.x, minimisation.fun
    

  def GetAndWriteBestFitToYaml(self, X_dps, initial_guess, row=None, filename="best_fit.yaml", minimisation_method="nominal", freeze={}):
    """
    Finds the best-fit parameters and writes them to a YAML file.

    Args:
        X (array): The independent variables.
        row (array): The row values.
        initial_guess (array): Initial values for the model parameters.
        wt (array): The weights for the data points (optional).
        filename (str): The name of the YAML file (default is "best_fit.yaml").
    """
    result = self.GetBestFit(X_dps, initial_guess, method=minimisation_method, freeze=freeze)
    self.best_fit = []
    ind = 0
    for col in self.Y_columns:
      if col in freeze.keys():
        self.best_fit.append(freeze[col])
      else:
        self.best_fit.append(result[0][ind])
        ind += 1

    self.best_fit_nll = result[1]
    dump = {
      "columns": self.Y_columns, 
      "best_fit": [float(i) for i in self.best_fit], 
      "best_fit_nll": float(self.best_fit_nll)
      }
    if row is not None:
      dump["row"] = [float(row.loc[0,col]) for col in self.Y_columns]
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)

  def GetAndWriteScanRangesToYaml(self, X_dps, col, row=None, filename="scan_ranges.yaml", estimated_sigmas_shown=3.2, estimated_sigma_step=0.4):
    """
    Computes the scan ranges for a given column and writes them to a YAML file.

    Args:
        X (array): The independent variables.
        row (array): The row values.
        col (str): The column name for the parameter to be scanned.
        wt (array): The weights for the data points (optional).
        filename (str): The name of the YAML file (default is "scan_ranges.yaml").
    """
    scan_values = self.GetScanXValues(X_dps, col, estimated_sigmas_shown=estimated_sigmas_shown, estimated_sigma_step=estimated_sigma_step)
    dump = {
      "columns" : self.Y_columns, 
      "varied_column" : col,
      "scan_values" : scan_values
    }
    if row is not None:
      dump["row"] = [float(row.loc[0,col]) for col in self.Y_columns]
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)

  def GetAndWriteScanToYaml(self, X_dps, col, col_val, row=None, freeze={}, filename="scan.yaml", minimisation_method="nominal"):

    col_index = self.Y_columns.index(col)
    Y = copy.deepcopy(self.best_fit)
    Y[col_index] = col_val
    scan_freeze = copy.deepcopy(freeze)    
    scan_freeze[col] = col_val

    if len(self.Y_columns) > 1 and not (len(list(scan_freeze.keys())) == len(self.Y_columns)):
      if self.verbose:
        print(f"Profiled fit for {col}={col_val}")
      result = self.GetBestFit(X_dps, pd.DataFrame([Y],columns=self.Y_columns, dtype=np.float64), method=minimisation_method, freeze=scan_freeze)
      dump = {
        "columns" : self.Y_columns, 
        "varied_column" : col,
        "nlls": [float(result[1] - self.best_fit_nll)],
        "profiled_columns" : [k for k in self.Y_columns if k != col],
        "profiled_values" : [float(i) for i in result[0]],
        "scan_values" : [float(col_val)],
      }
    else:
      result = self.Run(X_dps, Y, return_ln=True, multiply_by=-2)
      dump = {
        "columns" : self.Y_columns, 
        "varied_column" : col,
        "nlls": [float(result - self.best_fit_nll)],
        "scan_values" : [float(col_val)],
      }
    if row is not None:
      dump["row"] = [float(row.loc[0,col]) for col in self.Y_columns]
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)  