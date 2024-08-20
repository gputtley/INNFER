import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf

from functools import partial
from pprint import pprint
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
    self.seed = 42
    self.verbose = True

    # saved parameters
    self.best_fit = None
    self.best_fit_nll = None

  def _Gaussian(self, x):
    """
    Computes the Gaussian distribution.

    Args:
        x (float): The input value.

    Returns:
        float: The Gaussian distribution value.
    """
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * float(x)**2)

  def _LogNormal(self, x):
    """
    Computes the Log-Normal distribution.

    Args:
        x (float): The input value.

    Returns:
        float: The Log-Normal distribution value.
    """
    return 1 / (float(x) * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (np.log(float(x)))**2)

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

  def _SetSeed(self):
    tf.random.set_seed(self.seed)
    tf.keras.utils.set_random_seed(self.seed)
    np.random.seed(self.seed)

  def Run(self, inputs, Y, return_ln=True, multiply_by=1):
    """
    Evaluates the likelihood for given data.

    Args:
        inputs (array): The data processors for an unbinned/unbinned_extended fit or the bin values in the binned/binned_extended case.
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
      lkld_val = self.Unbinned(inputs, Y, return_ln=return_ln)
    elif self.type == "unbinned_extended":
      lkld_val = self.UnbinnedExtended(inputs, Y, return_ln=return_ln)
    elif self.type == "binned":
      lkld_val = self.Binned(inputs, Y, return_ln=return_ln)
    elif self.type == "binned_extended":
      lkld_val = self.BinnedExtended(inputs, Y, return_ln=return_ln)

    # Add constraint
    lkld_val += self._GetConstraint(Y)

    # End output
    end_time = time.time()
    if self.verbose:
      print(f"Y={Y.to_numpy().flatten()}, lnL={lkld_val}, time={round(end_time-start_time,2)}")

    return lkld_val*multiply_by

  def _LogSumExpTrick(self, ln_a, ln_b):
    mask = ln_b > ln_a
    ln_a[mask], ln_b[mask] = ln_b[mask], ln_a[mask]
    return ln_a + np.log1p(np.exp(ln_b - ln_a))

  def _GetYield(self, file_name, Y):
    if "yields" in self.models.keys():
      yd = self.models["yields"][file_name](Y)
    else:
      yd = 1.0
    return yd

  def _GetCombinedPDF(self, X, Y, wt_name=None, before_sum=False, normalise=True, gradient=[0], column_1=None, column_2=None):

    if isinstance(gradient, int):
      gradient = [gradient]

    first_loop = True
    sum_rate_params = 0.0

    # Loop through pdf models
    for name, pdf in self.models["pdfs"].items():

      # Get model scaling
      rate_param = self._GetYield(name, Y)
      if rate_param == 0: continue
      sum_rate_params += rate_param

      # Get rate times probability, get gradients simultaneously if required
      log_p_outputs = pdf.Probability(X.loc[:, self.data_parameters[name]["X_columns"]], Y, return_log_prob=True, order=gradient, column_1=column_1, column_2=column_2)
      ln_lklds_with_rate_params_outputs = [np.log(rate_param) + log_p for log_p in log_p_outputs]

      # Sum together probabilities of different files
      if first_loop:
        ln_lklds = [copy.deepcopy(ln_lklds_with_rate_params) for ln_lklds_with_rate_params in ln_lklds_with_rate_params_outputs]
        first_loop = False
      else:
        ln_lklds = [self._LogSumExpTrick(ln_lklds[ind],ln_lklds_with_rate_params) for ind, ln_lklds_with_rate_params in enumerate(ln_lklds_with_rate_params_outputs)]

    if normalise:
      # Rescale so total pdf integrate to 1
      ln_lklds = [ln_lklds_loop - np.log(sum_rate_params) for ln_lklds_loop in ln_lklds]

    # Return probability before sum
    if before_sum:
      if len(ln_lklds) == 1:
        return ln_lklds[0]
      else:
        return ln_lklds

    # Weight the events
    if wt_name is not None:
      ln_lklds = [X.loc[:,[wt_name]].to_numpy(dtype=np.float64)*ln_lklds_loop for ln_lklds_loop in ln_lklds]

    # Product the events
    ln_lkld = [np.sum(ln_lklds_loop, dtype=np.float128, axis=0) for ln_lklds_loop in ln_lklds]

    if len(ln_lkld) == 1:
      return ln_lkld[0]
    else:
      return ln_lkld


  def _GetConstraint(self, Y):

    # Add constraints
    ln_constraint = 0.0
    if "nuisance_constraints" in self.parameters.keys():
      for k, v in self.parameters["nuisance_constraints"].items():
        if k not in list(Y.columns): continue
        if v == "Gaussian":
          constraint = self._Gaussian(float(Y.loc[:,k].iloc[0]))
        elif v == "LogNormal":
          constraint = self._LogNormal(float(Y.loc[:,k].iloc[0]))
        ln_constraint += np.log(constraint)

    return ln_constraint


  def _CustomDPMethodForCombinedPDF(self, tmp, out, options={}):

    tmp_total = self._GetCombinedPDF(
      tmp, 
      options["Y"], 
      wt_name = options["wt_name"] if "wt_name" in options.keys() else None, 
      normalise = options["normalise"] if "normalise" in options.keys() else False, 
      gradient = options["gradient"] if "gradient" in options.keys() else [0], 
      column_1 = options["column_1"] if "column_1" in options.keys() else None,
      column_2 = options["column_2"] if "column_2" in options.keys() else None,
    )

    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      if isinstance(out, list):
        out = [out[i]+tmp_total[i] for i in range(len(out))]
      else:
        out += tmp_total
    return out

  def _CustomDPMethodForDMatrix(self, tmp, out, options={}):

    tmp_probs = self._GetCombinedPDF(
      tmp, 
      options["Y"], 
      wt_name = options["wt_name"] if "wt_name" in options.keys() else None, 
      normalise = options["normalise"] if "normalise" in options.keys() else False, 
      gradient = 1, 
      before_sum = True,
    )

    if out is None:
      out = np.zeros((len(options["scan_over"]), len(options["scan_over"])))

    for ind_1 in range(len(options["scan_over"])):
      for ind_2 in range(len(options["scan_over"])):
        out[ind_1, ind_2] += np.sum((tmp.loc[:,options["wt_name"]].to_numpy()**2) * tmp_probs[:, ind_1] * tmp_probs[:, ind_2], dtype=np.float64)

    return out

  def Binned(self, bin_values, Y, return_ln=True):

    # find the prediction
    predicted_bin_values = []
    for bin_index, bin_value in enumerate(bin_values):
      predicted_bin_values.append(np.sum([yield_functions[bin_index](Y) for yield_functions in self.models["bin_yields"].values()]))

    # normalise predicted_bin_values
    predicted_bin_values = [predicted_bin_value*(np.sum(bin_values))/float(np.sum(predicted_bin_values)) for predicted_bin_value in predicted_bin_values]

    # Get log likelihood
    ln_lkld = np.sum([(bin_values[bin_index]*np.log(predicted_bin_values[bin_index])) - predicted_bin_values[bin_index] for bin_index in range(len(bin_values))])

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

  def BinnedExtended(self, bin_values, Y, return_ln=True):

    ln_lkld = 0.0
    for bin_index, bin_value in enumerate(bin_values):
      predicted_bin_values = np.sum([yield_functions[bin_index](Y) for yield_functions in self.models["bin_yields"].values()])
      ln_lkld += (bin_value*np.log(predicted_bin_values)) - predicted_bin_values

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

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
    self._SetSeed()
    for dps_ind, X_dp in enumerate(X_dps):
      dps_ln_lkld = X_dp.GetFull(
        method = "custom",
        custom = self._CustomDPMethodForCombinedPDF,
        custom_options = {"Y" : Y, "wt_name" : X_dp.wt_name, "normalise" : normalise}
      )
      ln_lkld += dps_ln_lkld[0]

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


  def GetBestFit(self, X_dps, initial_guess, method="nominal", freeze={}, initial_step_size=0.05, save_best_fit=True):
    """
    Finds the best-fit parameters using numerical optimization.

    Args:
        X (array): The independent variables.
        initial_guess (array): Initial values for the model parameters.
        wts (array): The weights for the data points (optional).

    """
    if method in ["nominal","scipy"]:

      func_to_minimise = lambda Y: self.Run(X_dps, Y, multiply_by=-2)
      result = self.Minimise(func_to_minimise, initial_guess.to_numpy().flatten(), freeze=freeze, initial_step_size=initial_step_size)


    elif method == "scipy_with_gradients":

      func_val_and_jac = lambda Y: self.GetNLLWithGradient(X_dps, Y, multiply_by=-2)
      class NLLAndGradient():
        def __init__(self):
          self.jac = 0.0
          self.prev_nll_calc = False
        def GetNLL(self, Y):
          val, jac = func_val_and_jac(Y)
          self.jac = jac
          self.prev_nll_calc = True
          return val
        def GetJac(self,Y):
          if self.prev_nll_calc:
            self.prev_nll_calc = False
            return self.jac
          else:
            self.prev_nll_calc = False
            val, jac = func_val_and_jac(Y)
            return jac
      nllgrad = NLLAndGradient()
      result = self.Minimise(nllgrad.GetNLL, initial_guess.to_numpy().flatten(), freeze=freeze, initial_step_size=initial_step_size, method="scipy_with_gradients", jac=nllgrad.GetJac) # Add jacobian

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
      
    if save_best_fit:
      self.best_fit = result[0]
      self.best_fit_nll = result[1]

    return result

  def GetDMatrix(self, X_dps, scan_over):

    if self.type in ["unbinned","unbinned_extended"]:
      d_matrix = 0
      self._SetSeed()
      for dps_ind, X_dp in enumerate(X_dps):
        # Get D matrix
        dps_d_matrix = X_dp.GetFull(
          method = "custom",
          custom = self._CustomDPMethodForDMatrix,
          custom_options = {"Y" : pd.DataFrame([self.best_fit], columns=self.Y_columns), "wt_name" : X_dp.wt_name, "normalise" : True, "scan_over" : scan_over}
        )
        d_matrix += dps_d_matrix
    else:
      raise ValueError("D matrix only valid for unbinned fits.")

    # Extended fit information
    if self.type in ["unbinned_extended","binned_extended"]:
      # TO BE IMPLEMENTED: Add extended terms
      print() 

    # TO BE IMPLEMENTED: Add constraint terms  

    return d_matrix.tolist()

  def GetNLLWithGradient(self, X_dps, Y, multiply_by=1):

    start_time = time.time()

    # Check type of Y
    if not isinstance(Y, pd.DataFrame):
      Y = pd.DataFrame([Y], columns=self.Y_columns, dtype=np.float64)

    if self.type in ["unbinned","unbinned_extended"]:
      nll_gradient = [0,0]
      self._SetSeed()
      for dps_ind, X_dp in enumerate(X_dps):
        dps_nll_gradient = X_dp.GetFull(
          method = "custom",
          custom = self._CustomDPMethodForCombinedPDF,
          custom_options = {"Y" : Y, "wt_name" : X_dp.wt_name, "normalise" : True, "gradient" : [0,1]}
        )
        nll_gradient = [nll_gradient[0] + dps_nll_gradient[0], nll_gradient[1] + dps_nll_gradient[1]]
    else:
      # TO BE IMPLEMENTED: First derivative for binned fits
      print()

    # Extended fit information
    if self.type in ["unbinned_extended","binned_extended"]:
      # TO BE IMPLEMENTED: Add extended terms
      print() 

    # TO BE IMPLEMENTED: Add constraint terms  

    # End output
    end_time = time.time()
    if self.verbose:
      print(f"Y={Y.to_numpy().flatten()}, lnL={nll_gradient[0][0]}")
      print(f"Y={Y.to_numpy().flatten()}, dlnL/dY={nll_gradient[1]}, time={round(end_time-start_time,2)}")

    return multiply_by*nll_gradient[0], multiply_by*nll_gradient[1]

  def GetHessian(self, X_dps, column_1, column_2):

    if self.type in ["unbinned","unbinned_extended"]:
      second_derivative = 0
      self._SetSeed()
      for dps_ind, X_dp in enumerate(X_dps):
        # Get second derivative
        dps_second_derivative = X_dp.GetFull(
          method = "custom",
          custom = self._CustomDPMethodForCombinedPDF,
          custom_options = {"Y" : pd.DataFrame([self.best_fit], columns=self.Y_columns), "wt_name" : X_dp.wt_name, "normalise" : True, "gradient" : [2], "column_1" : column_1, "column_2" : column_2}
        )
        second_derivative += dps_second_derivative
    else:
      # TO BE IMPLEMENTED: Second derivative for binned fits
      print()

    # Extended fit information
    if self.type in ["unbinned_extended","binned_extended"]:
      # TO BE IMPLEMENTED: Add extended terms
      print() 

    # TO BE IMPLEMENTED: Add constraint terms  

    return -float(second_derivative)


  def GetApproximateUncertainty(self, X_dps, column, initial_step_fraction=0.001, min_step=0.1):

    # Find approximate value
    col_index = self.Y_columns.index(column)
    nll_could_be_neg = True

    while nll_could_be_neg:
      m1p1_vals = {}
      fin = True

      for sign in [1,-1]:

        # Get initial guess
        Y = copy.deepcopy(self.best_fit)
        step = max(self.best_fit[col_index]*initial_step_fraction,min_step)
        Y[col_index] = self.best_fit[col_index] + sign*step
        nll = self.Run(X_dps, Y, return_ln=True, multiply_by=-2) - self.best_fit_nll

        # Check if negative
        if nll < 0.0:
          self.best_fit[col_index] = Y[col_index]
          self.best_fit_nll += nll
          fin = False
          break

        # Get shift
        est_sig = np.sqrt(nll)
        m1p1_vals[sign] = step/est_sig

        # Test shift and reshuffle 
        Y[col_index] = self.best_fit[col_index] + sign*m1p1_vals[sign]
        nll = self.Run(X_dps, Y, return_ln=True, multiply_by=-2) - self.best_fit_nll
        est_sig = np.sqrt(nll)
        if nll > 0.0:
          m1p1_vals[sign] = m1p1_vals[sign]/est_sig

      if fin: 
        nll_could_be_neg = False

    return m1p1_vals


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
    m1p1_vals = self.GetApproximateUncertainty(X_dps, column, initial_step_fraction=initial_step_fraction, min_step=min_step)

    if self.verbose:
      print(f"Estimated result: {float(self.best_fit[col_index])} + {m1p1_vals[1]} - {m1p1_vals[-1]}")

    lower_scan_vals = [float(self.best_fit[col_index] - estimated_sigma_step*ind*m1p1_vals[-1]) for ind in range(int(np.ceil(estimated_sigmas_shown/estimated_sigma_step)),0,-1)]
    upper_scan_vals = [float(self.best_fit[col_index] + estimated_sigma_step*ind*m1p1_vals[1]) for ind in range(1,int(np.ceil(estimated_sigmas_shown/estimated_sigma_step))+1)]

    return lower_scan_vals + [float(self.best_fit[col_index])] + upper_scan_vals

  def Minimise(self, func, initial_guess, method="scipy", func_low_stat=None, freeze={}, initial_step_size=0.02, jac=None):
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
    
    elif method == "scipy_with_gradients":
      minimisation = minimize(func, initial_guess, jac=jac ,method='BFGS', tol=0.001, options={'gtol': 0.01, 'xrtol': 0.001})
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
    

  def GetAndWriteApproximateUncertaintyToYaml(self, X_dps, col, row=None, filename="approx_crossings.yaml"):  

    uncerts = self.GetApproximateUncertainty(X_dps, col)
    col_index = self.Y_columns.index(col)
    dump = {
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
      "columns" : self.Y_columns, 
      "varied_column" : col,
      "crossings" : {
        -2 : float(self.best_fit[col_index] - 2*uncerts[-1]),
        -1 : float(self.best_fit[col_index] - 1*uncerts[-1]),
        0  : float(self.best_fit[col_index]),
        1  : float(self.best_fit[col_index] + 1*uncerts[1]),
        2  : float(self.best_fit[col_index] + 2*uncerts[1]),
      }
    }
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)


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
      "best_fit_nll": float(self.best_fit_nll),
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
      }
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)

  def GetAndWriteCovarianceToYaml(self, hessian, row=None, scan_over=None, D_matrix=None, filename="covariance.yaml"):

    hessian = np.array(hessian)
    inverse_hessian = np.linalg.inv(hessian)

    if D_matrix is None:
      covariance = inverse_hessian.tolist()
    else:
      print(inverse_hessian)
      print(D_matrix)
      covariance = (inverse_hessian @ D_matrix @ inverse_hessian).tolist()

    dump = {
      "columns": self.Y_columns, 
      "matrix_columns" : scan_over, 
      "covariance" : covariance,
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
    }
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)

  def GetAndWriteDMatrixToYaml(self, X_dps, row=None, freeze={}, scan_over=None, filename="dmatrix.yaml"):
    """
    Finds the best-fit parameters and writes them to a YAML file.

    Args:
        X (array): The independent variables.
        row (array): The row values.
        initial_guess (array): Initial values for the model parameters.
        wt (array): The weights for the data points (optional).
        filename (str): The name of the YAML file (default is "best_fit.yaml").
    """
    if scan_over is None:
      scan_over = self.Y_columns
    scan_over = [col for col in scan_over if col not in freeze.keys()]

    D_matrix = self.GetDMatrix(X_dps, scan_over)

    dump = {
      "columns": self.Y_columns, 
      "matrix_columns" : scan_over, 
      "D_matrix" : D_matrix,
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
    }
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)


  def GetAndWriteHessianToYaml(self, X_dps, row=None, freeze={}, scan_over=None, filename="hessian.yaml"):
    """
    Finds the best-fit parameters and writes them to a YAML file.

    Args:
        X (array): The independent variables.
        row (array): The row values.
        initial_guess (array): Initial values for the model parameters.
        wt (array): The weights for the data points (optional).
        filename (str): The name of the YAML file (default is "best_fit.yaml").
    """
    if scan_over is None:
      scan_over = self.Y_columns
    scan_over = [col for col in scan_over if col not in freeze.keys()]

    # Get hessian entries
    hessian = np.zeros((len(scan_over),len(scan_over)))
    for index_1, column_1 in enumerate(scan_over):
      for index_2, column_2 in enumerate(scan_over):
        if index_1 <= index_2:
          hessian[index_1, index_2] = copy.deepcopy(self.GetHessian(X_dps, column_1, column_2))
          if index_1 != index_2:
            hessian[index_2, index_1] = copy.deepcopy(self.GetHessian(X_dps, column_1, column_2))
    hessian = hessian.tolist()

    dump = {
      "columns": self.Y_columns, 
      "matrix_columns" : scan_over, 
      "hessian" : hessian,
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
    }
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
      "scan_values" : scan_values,
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
    }
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
      result = self.GetBestFit(X_dps, pd.DataFrame([Y],columns=self.Y_columns, dtype=np.float64), method=minimisation_method, freeze=scan_freeze, save_best_fit=False)
      dump = {
        "columns" : self.Y_columns, 
        "varied_column" : col,
        "nlls": [float(result[1] - self.best_fit_nll)],
        "profiled_columns" : [k for k in self.Y_columns if k != col],
        "profiled_values" : [float(i) for i in result[0]],
        "scan_values" : [float(col_val)],
        "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
      }
    else:
      result = self.Run(X_dps, Y, return_ln=True, multiply_by=-2)
      dump = {
        "columns" : self.Y_columns, 
        "varied_column" : col,
        "nlls": [float(result - self.best_fit_nll)],
        "scan_values" : [float(col_val)],
        "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
      }
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)