import copy
import gc
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf

from functools import partial
from iminuit import Minuit
from pprint import pprint
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

from minimise import Minimise as CustomMinimise
from useful_functions import MakeDirectories

class Likelihood():
  """
  A class representing a likelihood function.
  """
  def __init__(self, models, likelihood_type="unbinned_extended", constraints=[], constraint_center=None, X_columns=[], Y_columns=[], Y_columns_per_model={}, categories=None):
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
    self.X_columns = X_columns
    self.Y_columns = Y_columns
    self.Y_columns_per_model = Y_columns_per_model
    self.categories = categories

    self.seed = 42
    self.verbose = True
    self.constraints = constraints
    self.constraint_center = constraint_center

    # saved parameters
    self.best_fit = None
    self.best_fit_nll = None
    self.minimisation_step = None
    self.yield_splines_value = None
    self.yield_splines = None
    self.hessian = None
    self.hessian_columns = None
    self.D_matrix = None
    self.D_matrix_columns = None
    self.covariance = None


  def _CheckLogProbsForNaNs(self, log_probs, gradient=[0]):

    # Find nans and infs
    for k, v_arr in log_probs.items():
      for v_ind, v in enumerate(v_arr):

        if np.isnan(v.flatten()).any():
          nan_indices = [i[0] for i in np.where(np.isnan(v.flatten()))]

          for ind in nan_indices:

            zero_everywhere = True
            for _, v_c_arr in log_probs.items():
              if v_c_arr[v_ind].flatten()[ind] != 0:
                zero_everywhere = False
                break 

            if zero_everywhere:
              raise ValueError(f"NaN found in model for {k}. This is likely due to a zero probability density function. Please check your model.")
            else:
              if gradient[v_ind] == 0:
                log_probs[k][v_ind][nan_indices,:] = -np.inf
              else:
                log_probs[k][v_ind][nan_indices,:] = np.inf

    return log_probs
  

  def _ConstraintGaussian(self, x, derivative=0):
    """
    Computes the Gaussian distribution.

    Args:
        x (float): The input value.

    Returns:
        float: The Gaussian distribution value.
    """
    x = float(x)
    gauss =  1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
    if derivative == 0:
      return gauss
    elif derivative == 1:
      return -x * gauss
    elif derivative == 2:
      return (x**2 - 1) * gauss


  def _CustomDPMethod(self, tmp, out, options={}):

    tmp_total = options["function"](tmp, options["Y"], **options["options"])

    if out is None:
      out = copy.deepcopy(tmp_total)
    else:
      if isinstance(out, list):
        out = [out[i]+tmp_total[i] for i in range(len(out))]
      else:
        out += tmp_total
    return out


  def _CustomDPMethodForDMatrix(self, tmp, out, options={}):


    # Get the per event log likelihoods
    if self.type == "unbinned":
      # For unbinned
      tmp_probs = self._GetEventLoopUnbinned(
        tmp, 
        options["Y"], 
        wt_name = options["wt_name"] if "wt_name" in options.keys() else None, 
        gradient = [1], 
        column_1 = self.Y_columns,
        before_sum = True,
        category = options["category"] if "category" in options.keys() else None
      )[0]
    elif self.type == "unbinned_extended":
      # For unbinned extended
      tmp_probs = self._GetEventLoopUnbinnedExtended(
        tmp, 
        options["Y"], 
        wt_name = options["wt_name"] if "wt_name" in options.keys() else None, 
        gradient = [1], 
        column_1 = self.Y_columns,
        before_sum = True,
        category = options["category"] if "category" in options.keys() else None
      )[0]
      # Add in the extended part
      for name in self.models["yields"].keys():
        tmp_probs -= self._GetYieldGradient(name, options["Y"], gradient=1, column_1=self.Y_columns, category=options["category"])/options["sum_wts"]

    # Add constraints
    tmp_probs += self._GetConstraint(options["Y"], derivative=1, column_1=self.Y_columns)/options["sum_wts"]

    # Check for nans
    if np.isnan(tmp_probs).any():
      nan_rows_mask = np.isnan(tmp_probs).any(axis=1)  # True for rows with NaNs
      tmp_probs[nan_rows_mask] = 0.0

    # Initiate D matrix
    if out is None:
      out = np.zeros((len(options["scan_over"]), len(options["scan_over"])))

    # Check for constant values of first derivatives
    consts = []
    for ind in range(len(options["scan_over"])):
      if len(np.unique(tmp_probs[:,self.Y_columns.index(options["scan_over"][ind])].astype(np.float32))) == 1:
        consts.append(True)
      else:
        consts.append(False)

    # Calculate D matrix
    for ind_1 in range(len(options["scan_over"])):
      for ind_2 in range(len(options["scan_over"])):
        if consts[ind_1] or consts[ind_2]:
          out[ind_1, ind_2] = np.nan
        elif out[ind_1, ind_2] is not np.nan:
          out[ind_1, ind_2] += np.sum((tmp.loc[:,options["wt_name"]].to_numpy()**2) * tmp_probs[:, self.Y_columns.index(options["scan_over"][ind_1])] * tmp_probs[:, self.Y_columns.index(options["scan_over"][ind_2])], dtype=np.float64)

    return out

  def _GetBinned(self, bin_values, Y, category=None):

    predicted_bin_values = [np.sum([yield_functions[bin_index](Y) for yield_functions in self.models["bin_yields"][category].values()]) for bin_index, bin_value in enumerate(bin_values)]
    non_zero_indices = np.array(predicted_bin_values) > 0
    predicted_bin_values = list(np.array(predicted_bin_values)[non_zero_indices])
    bin_values = list(np.array(bin_values)[non_zero_indices])
    predicted_bin_values = [predicted_bin_value*(np.sum(bin_values))/float(np.sum(predicted_bin_values)) for predicted_bin_value in predicted_bin_values]
    ln_lkld = np.sum([(bin_value*np.log(predicted_bin_values[bin_index])) - predicted_bin_values[bin_index] for bin_index, bin_value in enumerate(bin_values)])

    return ln_lkld

  def _GetBinnedExtended(self, bin_values, Y, category=None):

    predicted_bin_values = [np.sum([yield_functions[bin_index](Y) for yield_functions in self.models["bin_yields"][category].values()]) for bin_index, bin_value in enumerate(bin_values)]
    non_zero_indices = np.array(predicted_bin_values) > 0
    predicted_bin_values = list(np.array(predicted_bin_values)[non_zero_indices])
    bin_values = list(np.array(bin_values)[non_zero_indices])
    ln_lkld = np.sum([(bin_value*np.log(predicted_bin_values[bin_index])) - predicted_bin_values[bin_index] for bin_index, bin_value in enumerate(bin_values)])

    return ln_lkld

  def _GetConstraint(self, Y, derivative=0, column_1=None, column_2=None):

    first_loop = True
    ln_constraint = 0.0
    for k in self.constraints:

      if k not in list(Y.columns): continue

      constraint_func = self._ConstraintGaussian

      constraint_input = float(Y.loc[:,k].iloc[0])
      if self.constraint_center is not None:
        if k in self.constraint_center.columns:
          constraint_input -= float(self.constraint_center.loc[:,k].iloc[0])

      # Get nominal value
      if derivative == 0:

        val = np.log(constraint_func(constraint_input, derivative=0))

        if first_loop:
          ln_constraint = val
          first_loop = False
        else:
          ln_constraint += val

      # Get first derivative
      elif derivative == 1:

        vals = np.array([])
        for col in column_1:

          if col != k:
            val = 0.0
          else:
            val = (1/constraint_func(constraint_input, derivative=0))*constraint_func(constraint_input, derivative=1)

          vals = np.append(vals, val)

        if first_loop:
          ln_constraint = vals
          first_loop = False
        else:
          ln_constraint += vals

      elif derivative == 2:

        if first_loop:
          ln_constraint = 0.0
          first_loop = False
      
        if column_1[0] == column_2[0] and column_1[0] == k:
          val = -(constraint_func(constraint_input, derivative=1)/constraint_func(constraint_input, derivative=0))**2 + (constraint_func(constraint_input, derivative=2)/constraint_func(constraint_input, derivative=0))
          ln_constraint += val
          
    return ln_constraint


  def _GetLogProbs(self, X, Y, gradient=0, column_1=None, column_2=None, category=None):

    # Initiate log probs
    log_probs = {}

    if category is None or category not in self.models["pdfs"].keys():
      raise ValueError("Category must be specified and must be in the models dictionary.")

    # Loop through pdf models
    for name, pdf in self.models["pdfs"][category].items():

      # Get model scaling
      rate_param = self._GetYield(name, Y, category=category)
      if rate_param == 0: continue

      # Get rate times probability, get gradients simultaneously if required
      log_probs[name] = pdf.Probability(
        X.loc[:,self.X_columns], 
        Y.loc[:,self.Y_columns], 
        return_log_prob=True, 
        order=gradient, 
        column_1=column_1, 
        column_2=column_2
      )

      # Shift density with regression
      log_probs[name] = self._ShiftDensityByRegression(
        log_probs[name],
        X.loc[:,self.X_columns], 
        Y.loc[:,self.Y_columns],
        name,
        gradient=gradient, 
        column_1=column_1, 
        column_2=column_2,
        category=category        
      )

    # check for nans
    log_probs = self._CheckLogProbsForNaNs(log_probs, gradient=gradient)

    return log_probs


  def _GetH1(self, log_probs, Y, log=False, category=None):

    # Loop through pdf models
    first_loop = True
    for name, _ in self.models["pdfs"][category].items():

      # Get model scaling
      rate_param = self._GetYield(name, Y, category=category)

      if rate_param == 0: continue

      # Get rate times probability
      ln_lklds_with_rate_params = np.log(rate_param) + log_probs[name]

      # Sum together probabilities of different files
      if first_loop:
        H1 = copy.deepcopy(ln_lklds_with_rate_params)
        first_loop = False
      else:
        H1 = self._HelperLogSumExpTrick(H1,ln_lklds_with_rate_params)

    # Change to non log
    if not log:
      H1 = np.exp(H1)

    return H1


  def _GetH1FirstDerivative(self, log_probs, log_probs_first_derivative, Y, column_1=None, category=None):

    # Loop through pdf models
    first_loop = True
    for name, _ in self.models["pdfs"][category].items():

      # Get rate times probability
      ln_lklds_with_rate_params = np.exp(log_probs[name]) * (self._GetYieldGradient(name, Y, gradient=1, column_1=column_1, category=category) + (self._GetYield(name, Y, category=category) * log_probs_first_derivative[name]))

      # Sum together probabilities of different files
      if first_loop:
        H1FirstDerivative = copy.deepcopy(ln_lklds_with_rate_params)
        first_loop = False
      else:
        H1FirstDerivative += ln_lklds_with_rate_params

    return H1FirstDerivative


  def _GetH1SecondDerivative(self, log_probs, log_probs_first_derivative_1, log_probs_first_derivative_2, log_probs_second_derivative, Y, column_1=None, column_2=None, category=None):

    # Loop through pdf models
    first_loop = True
    for name, _ in self.models["pdfs"][category].items():

      # Get rate times probability
      ln_lklds_with_rate_params = np.exp(log_probs[name]) * (self._GetYieldGradient(name, Y, gradient=2, column_1=column_1, column_2=column_2, category=category) + (self._GetYieldGradient(name, Y, gradient=1, column_1=column_1, category=category) * log_probs_first_derivative_2[name]) + (self._GetYieldGradient(name, Y, gradient=1, column_1=column_2, category=category) * log_probs_first_derivative_1[name]) + (self._GetYield(name, Y, category=category) * (log_probs_second_derivative[name] + (log_probs_first_derivative_1[name]*log_probs_first_derivative_2[name]))))

      # Sum together probabilities of different files
      if first_loop:
        H1SecondDerivative = copy.deepcopy(ln_lklds_with_rate_params)
        first_loop = False
      else:
        H1SecondDerivative += ln_lklds_with_rate_params

    return H1SecondDerivative


  def _GetH2(self, Y, log=False, category=None):

    # Loop through pdf models
    H2 = 0.0
    for name, _ in self.models["pdfs"][category].items():

      # Get model scaling
      rate_param = self._GetYield(name, Y, category=category)
      H2 += rate_param

    # Change to log
    if log:
      H2 = np.log(H2)

    return H2


  def _GetH2FirstDerivative(self, Y, column_1=None, category=None):

    # Loop through pdf models
    H2FirstDerivative = 0.0
    for name, pdf in self.models["pdfs"][category].items():

      # Get model scaling
      rate_param = self._GetYieldGradient(name, Y, gradient=1, column_1=column_1, category=category)
      H2FirstDerivative += rate_param

    return H2FirstDerivative


  def _GetH2SecondDerivative(self, Y, column_1=None, column_2=None, category=None):

    # Loop through pdf models
    H2SecondDerivative = 0.0
    for name, pdf in self.models["pdfs"][category].items():

      # Get model scaling
      rate_param = self._GetYieldGradient(name, Y, gradient=2, column_1=column_1, column_2=column_2, category=category)
      H2SecondDerivative += rate_param

    return H2SecondDerivative


  def _GetEventLoopUnbinned(self, X, Y, wt_name=None, gradient=[0], column_1=None, column_2=None, before_sum=False, category=None):

    # Get probabilities and other first derivative if needed
    if 2 in gradient:
      get_log_prob_gradients = [0,1,2]
    elif 1 in gradient:
      get_log_prob_gradients = [0,1]
    else:
      get_log_prob_gradients = [0]
    log_probs = self._GetLogProbs(X, Y, gradient=get_log_prob_gradients, column_1=column_1, column_2=column_2, category=category)

    if 2 in gradient:
      if column_1 != column_2:
        first_derivative_other = self._GetLogProbs(X, Y, gradient=1, column_1=column_2, category=category)
      else:
        first_derivative_other = {k:v[get_log_prob_gradients.index(1)] for k,v in log_probs.items()}

    # Get H1 and H2
    log_H1 = self._GetH1({k:v[get_log_prob_gradients.index(0)] for k,v in log_probs.items()}, Y, log=True, category=category)
    log_H2 = self._GetH2(Y, log=True, category=category)
    if 1 in get_log_prob_gradients:
      H1_grad_1_col_1 = self._GetH1FirstDerivative({k:v[get_log_prob_gradients.index(0)] for k,v in log_probs.items()}, {k:v[get_log_prob_gradients.index(1)] for k,v in log_probs.items()}, Y, column_1=column_1, category=category)
      H2_grad_1_col_1 = self._GetH2FirstDerivative(Y, column_1=column_1, category=category)
    if 2 in get_log_prob_gradients:
      H1_grad_2 = self._GetH1SecondDerivative({k:v[get_log_prob_gradients.index(0)] for k,v in log_probs.items()}, {k:v[get_log_prob_gradients.index(1)] for k,v in log_probs.items()}, first_derivative_other, {k:v[get_log_prob_gradients.index(2)] for k,v in log_probs.items()}, Y, column_1=column_1, column_2=column_2, category=category)
      H2_grad_2 = self._GetH2SecondDerivative(Y, column_1=column_1, column_2=column_2, category=category)
      H1_grad_1_col_2 = self._GetH1FirstDerivative({k:v[get_log_prob_gradients.index(0)] for k,v in log_probs.items()}, first_derivative_other, Y, column_1=column_2, category=category)
      H2_grad_1_col_2 = self._GetH2FirstDerivative(Y, column_1=column_2, category=category)
    
    # Calculate and weight sum loop terms
    ln_lkld = []
    for grad in gradient:

      # Calculate loop terms
      if grad == 0:
        ln_lklds = log_H1 - log_H2
      elif grad == 1:
        ln_lklds = self._HelperLogMultiplyTrick(-log_H1, H1_grad_1_col_1) - ((1/np.exp(log_H2))*H2_grad_1_col_1)
      elif grad == 2:
        term_1 = self._HelperLogMultiplyTrick(-log_H1, H1_grad_2)
        term_2 = -self._HelperLogMultiplyTrick(-2*log_H1, H1_grad_1_col_1*H1_grad_1_col_2)
        term_3 = -np.exp(-log_H2)*H2_grad_2
        term_4 = np.exp(-2*log_H2)*H2_grad_1_col_1*H2_grad_1_col_2
        ln_lklds = term_1 + term_2 + term_3 + term_4

      if before_sum:

        # Append full array
        ln_lkld.append(copy.deepcopy(ln_lklds))

      else:

        if np.isnan(ln_lklds).any():
          nan_rows_mask = np.isnan(ln_lklds).any(axis=1)  # True for rows with NaNs
          ln_lklds[nan_rows_mask] = 0.0

        # Weight the events
        if wt_name is not None:
          ln_lklds = X.loc[:,[wt_name]].to_numpy(dtype=np.float64)*ln_lklds
      
        # Product the events
        sum_ln_lkld = np.sum(ln_lklds, dtype=np.float128, axis=0)
        ln_lkld += [sum_ln_lkld if len(sum_ln_lkld) > 1 else sum_ln_lkld[0]]

    return ln_lkld


  def _GetEventLoopUnbinnedExtended(self, X, Y, wt_name=None, gradient=[0], column_1=None, column_2=None, before_sum=False, category=None):

    # Get probabilities and other first derivative if needed
    if 2 in gradient:
      get_log_prob_gradients = [0,1,2]
    elif 1 in gradient:
      get_log_prob_gradients = [0,1]
    else:
      get_log_prob_gradients = [0]
    log_probs = self._GetLogProbs(X, Y, gradient=get_log_prob_gradients, column_1=column_1, column_2=column_2, category=category)
    if 2 in gradient:
      first_derivative_other = self._GetLogProbs(X, Y, gradient=1, column_1=column_2, category=category)

    # Get H1 and H2
    log_H1 = self._GetH1({k:v[get_log_prob_gradients.index(0)] for k,v in log_probs.items()}, Y, log=True, category=category)
    if 1 in get_log_prob_gradients:
      H1_grad_1_col_1 = self._GetH1FirstDerivative({k:v[get_log_prob_gradients.index(0)] for k,v in log_probs.items()}, {k:v[get_log_prob_gradients.index(1)] for k,v in log_probs.items()}, Y, column_1=column_1, category=category)
    if 2 in get_log_prob_gradients:
      H1_grad_2 = self._GetH1SecondDerivative({k:v[get_log_prob_gradients.index(0)] for k,v in log_probs.items()}, {k:v[get_log_prob_gradients.index(1)] for k,v in log_probs.items()}, first_derivative_other, {k:v[get_log_prob_gradients.index(2)] for k,v in log_probs.items()}, Y, column_1=column_1, column_2=column_2, category=category)
      H1_grad_1_col_2 = self._GetH1FirstDerivative({k:v[get_log_prob_gradients.index(0)] for k,v in log_probs.items()}, first_derivative_other, Y, column_1=column_2, category=category)

    # Calculate and weight sum loop terms
    ln_lkld = []
    for grad in gradient:

      # Calculate loop terms
      if grad == 0:
        ln_lklds = log_H1
      elif grad == 1:
        ln_lklds = self._HelperLogMultiplyTrick(-log_H1, H1_grad_1_col_1)
      elif grad == 2:
        ln_lklds = self._HelperLogMultiplyTrick(-log_H1, H1_grad_2) - self._HelperLogMultiplyTrick(-2*log_H1, H1_grad_1_col_1*H1_grad_1_col_2)

      if before_sum:
        # Append full array
        ln_lkld.append(copy.deepcopy(ln_lklds))
      else:
        # Weight the events
        if wt_name is not None:
          ln_lklds = X.loc[:,[wt_name]].to_numpy(dtype=np.float64)*ln_lklds
        # Product the events
        sum_ln_lkld = np.sum(ln_lklds, dtype=np.float128, axis=0)
        ln_lkld += [sum_ln_lkld if len(sum_ln_lkld) > 1 else sum_ln_lkld[0]]

    return ln_lkld


  def _GetYield(self, file_name, Y, category=None):

    yd = 1.0
    if "yields" in self.models.keys():
      if category in self.models["yields"].keys():
        if file_name in self.models["yields"][category].keys():
          yd = self.models["yields"][category][file_name](Y)
      
    return yd


  def _GetYieldGradient(self, file_name, Y, gradient=0, column_1=None, column_2=None, from_spline=False, category=None):

    if gradient == 0: 
      return self._GetYield(file_name, Y, category=category)
    elif gradient == 1:
      if not from_spline:
        return self._HelperNumericalGradientFromLinear(partial(self._GetYield, file_name, category=category), Y, column_1, file_name, gradient=1)
      else:
        return self._HelperNumericalGradientFromSpline(partial(self._GetYield, file_name, category=category), Y, column_1, file_name, gradient=1)
    elif gradient == 2:
      if not from_spline:
        return self._HelperNumericalGradientFromLinear(partial(self._GetYield, file_name, category=category), Y, column_1, file_name, gradient=2)[0]
      else:
        if column_1 is None and column_2 is None:
          raise ValueError("You need to specify column_1 and column_2 to get the second derivative.")
        if column_1 == column_2:
          return self._HelperNumericalGradientFromSpline(partial(self._GetYield, file_name, category=category), Y, column_1, file_name, gradient=2)[0]
        else:
          inner_func = lambda func, column, file_name, val, : self._HelperNumericalGradientFromSpline(func, val, column, file_name, gradient=1)
          inner_func = partial(inner_func, partial(self._GetYield, file_name, category=category), column_1, file_name)
          return self._HelperNumericalGradientFromSpline(inner_func, Y, column_2, file_name, gradient=1)[0]


  def _HelperFreeze(self, freeze, initial_guess, func, jac=None, non_list_input=False):

    if len(list(freeze.keys())) > 0:
      initial_guess_before = copy.deepcopy(initial_guess)
      initial_guess = [initial_guess_before[ind] for ind, col in enumerate(self.Y_columns) if col not in freeze.keys()]

      def updated_func(x, unpack=False):
        extended_x = []
        x_ind = 0
        for col in self.Y_columns:
          if col in freeze.keys():
            extended_x.append(freeze[col])
          else:
            extended_x.append(x[x_ind])
            x_ind += 1
        if unpack:
          return func(*extended_x)
        else:
          return func(extended_x)

      if jac is not None:

        def updated_jac(x, unpack=False):
          extended_x = []
          return_indices = []
          x_ind = 0
          for col_ind, col in enumerate(self.Y_columns):
            if col in freeze.keys():
              extended_x.append(freeze[col])
            else:
              extended_x.append(x[x_ind])
              x_ind += 1
              return_indices.append(col_ind)
          if unpack:
            return jac(*extended_x)[return_indices]
          else:
            return jac(extended_x)[return_indices]
          
      if non_list_input:

        inputs = ",".join(['p'+str(ind) for ind in range(len(initial_guess))])
        func_non_list = eval(f"lambda {inputs}: updated_func(np.array([{inputs}]), unpack=True)", {"updated_func": updated_func, 'np': np})
        if jac is not None:
          jac_non_list = eval(f"lambda {inputs}: updated_jac(np.array([{inputs}]), unpack=True)", {"updated_jac": updated_jac, 'np': np})

        if jac is None:
          return func_non_list, initial_guess
        else:
          return func_non_list, jac_non_list, initial_guess

      else:

        if jac is None:
          return updated_func, initial_guess
        else:
          return updated_func, updated_jac, initial_guess


    if jac is None:
      return func, initial_guess
    else:
      return func, jac, initial_guess


  def _HelperSetSeed(self):
    tf.random.set_seed(self.seed)
    tf.keras.utils.set_random_seed(self.seed)
    np.random.seed(self.seed)


  def _HelperLogSumExpTrick(self, ln_a, ln_b):
    mask = ln_b > ln_a
    ln_a[mask], ln_b[mask] = ln_b[mask], ln_a[mask]
    return ln_a + np.log1p(np.exp(ln_b - ln_a))


  def _HelperLogMultiplyTrick(self, ln_a, b):

    if ln_a.shape[1] == 1 and b.shape[1] > 1:
      ln_a = np.repeat(ln_a, b.shape[1], axis=1)
    elif b.shape[1] == 1 and ln_a.shape[1] > 1:
      b = np.repeat(b, ln_a.shape[1], axis=1)

    indices = (b!=0)
    b_sign = np.sign(b)
    out = np.zeros(b.shape)
    out[indices] = b_sign[indices]*np.exp(ln_a[indices] + np.log(np.abs(b[indices])))
    return out


  def _HelperNumericalGradientFromLinear(self, func, val, column, file_name, gradient=1, shift=1e-5):

    if column is None:
      columns = list(val.columns)
    elif isinstance(column, str):
      columns = [column]
    else:
      columns = column

    if gradient == 2:
      return [0.0]*len(columns)

    grads = []
    for col in columns:
      diff_value = float(val.loc[0,col])
      up_test_value = copy.deepcopy(val)
      up_test_value.loc[0,col] = diff_value + shift
      down_test_value = copy.deepcopy(val)
      down_test_value.loc[0,col] = diff_value - shift 
      grads.append((func(up_test_value)-func(down_test_value))/(2*shift))

    return np.array(grads)


  def _HelperNumericalGradientFromSpline(self, func, val, column, file_name, n_points=10, frac=0.01, gradient=1):

    if column is None:
      columns = list(val.columns)
    elif isinstance(column, str):
      columns = [column]
    else:
      columns = column

    grads = []

    # Check if spline stored
    if self.yield_splines is None:
      fit_spline = True
    elif not val.equals(self.yield_splines_value):
      fit_spline = True
    elif file_name not in self.yield_splines.keys():
      fit_spline = True
    else:
      fit_spline = False

    # loop through columns
    splines = {}
    for col in columns:

      # Value of columN to differentiate
      diff_value = float(val.loc[0,col])
      if diff_value < 1:
        diff_range = np.linspace(-1, 1, 10)
      else:
        diff_range = np.linspace(diff_value*(1-frac), diff_value*(1+frac), n_points)

      # Set up spline input
      x = []
      y = []
      for i in diff_range:
        test_value = copy.deepcopy(val)
        test_value.loc[0,col] = i
        y_res = func(test_value)
        if y_res is not None and not np.isnan(y_res):
          x.append(i)
          y.append(y_res)

      # If empty return None
      if len(y) <= 3:
        grads.append(np.nan)
        continue

      if fit_spline:
        # fit spline
        splines[col] = UnivariateSpline(x, y, s=3)
        # Get derivative
        derivative = splines[col].derivative(gradient)
      else:
        # Get derivative
        derivative = self.yield_splines[file_name][col].derivative(gradient)

      # Return value at point
      grads.append(derivative(diff_value))

    if fit_spline:
      if self.yield_splines is None:
        self.yield_splines = {}
      self.yield_splines_value = copy.deepcopy(val)
      self.yield_splines[file_name] = copy.deepcopy(splines)

    return np.array(grads)


  def _ShiftDensityByRegression(self, log_probs, X, Y, file_name, gradient=0, column_1=None, column_2=None, category=None):

    if "pdf_shifts_with_regression" not in self.models.keys():
      return log_probs

    # Make gradient loop
    if not isinstance(gradient, list):
      gradient_loop = [gradient]
      log_probs = [log_probs]
    else:
      gradient_loop = gradient

    # Make combined dataset
    combined = pd.concat([X, pd.DataFrame([Y.iloc[0]] * len(X)).reset_index(drop=True)], axis=1)
    del X, Y
    gc.collect()

    # Loop through regression models
    if file_name in self.models["pdf_shifts_with_regression"][category].keys():
      for k, v in self.models["pdf_shifts_with_regression"][category][file_name].items():

        # Get predictions
        if max(gradient_loop) == 0:
          get_gradient = [0]
        elif max(gradient_loop) == 1:
          get_gradient = [0,1]
        elif max(gradient_loop) == 2:
          get_gradient = [0,1,2]

        pred = v.Predict(combined, order=get_gradient, column_1=column_1, column_2=column_2)

        # Update log prob
        for ind, grad in enumerate(gradient_loop):
          if grad == 0:
            log_probs[ind] += np.log(pred[0])                
          elif grad == 1:
            log_probs[ind] += pred[1]/pred[0]
          elif grad == 2:
            log_probs[ind] += (pred[2]/pred[0]) - (pred[1]/pred[0])**2  

        # Normalise predictions
        for ind, grad in enumerate(gradient_loop):
          if "pdf_shifts_with_regression_norm_spline" in self.models.keys():
            if file_name in self.models["pdf_shifts_with_regression_norm_spline"][category].keys():
              if k in self.models["pdf_shifts_with_regression_norm_spline"][category][file_name].keys():
                norm = self.models["pdf_shifts_with_regression_norm_spline"][category][file_name][k](combined.loc[:,[k]])
                if grad == 0:
                  log_probs[ind] += np.log(norm)                
                elif grad == 1 and k in column_1:
                  norm_grad_1 = self.models["pdf_shifts_with_regression_norm_spline"][category][file_name][k].derivative(1)(combined.loc[:,[k]])
                  log_probs[ind][:, [column_1.index(k)]] += norm_grad_1/norm
                elif grad == 2 and column_1 == k and column_2 == k:
                  norm_grad_1 = self.models["pdf_shifts_with_regression_norm_spline"][category][file_name][k].derivative(1)(combined.loc[:,[k]])
                  norm_grad_2 = self.models["pdf_shifts_with_regression_norm_spline"][category][file_name][k].derivative(2)(combined.loc[:,[k]])
                  log_probs[ind] += (norm_grad_2/norm) - (norm_grad_1/norm)**2
        
    if not isinstance(gradient, list):
      log_probs = log_probs[0]

    return log_probs


  def Run(self, inputs, Y, multiply_by=1, gradient=0, column_1=None, column_2=None, numerical_gradient=False):
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

    # Check type of gradient
    convert_back = False
    if isinstance(gradient,int):
      gradient = [gradient]
      convert_back = True
    if 2 in gradient and column_1 is None and column_2 is None:
      raise ValueError("You must specifiy the exact columns required for the second derivative.")
    if 1 in gradient and column_1 is None:
      column_1 = self.Y_columns
    if isinstance(column_1, str):
      column_1 = [column_1]
    if isinstance(column_2, str):
      column_2 = [column_2]  

    # Do numerical gradients
    if numerical_gradient and gradient != [0]:
      raise NotImplementedError("Numerical gradients are not implemented yet for this class. Please use analytical gradients or scans.")

    # Run likelihood
    first = True
    for category in self.categories:

      if self.type == "unbinned":
        cat_val = self.LikelihoodUnbinned(inputs[category], Y, gradient=gradient, column_1=column_1, column_2=column_2, category=category)
      elif self.type == "unbinned_extended":
        cat_val = self.LikelihoodUnbinnedExtended(inputs[category], Y, gradient=gradient, column_1=column_1, column_2=column_2, category=category)
      elif self.type == "binned":
        cat_val = self.LikelihoodBinned(inputs[category], Y, gradient=gradient, column_1=column_1, column_2=column_2, category=category)
      elif self.type == "binned_extended":
        cat_val = self.LikelihoodBinnedExtended(inputs[category], Y, gradient=gradient, column_1=column_1, column_2=column_2, category=category)
      if first:
        lkld_val = copy.deepcopy(cat_val)
        first = False
      else:
        for grad_ind, grad in enumerate(gradient):
          lkld_val[grad_ind] += cat_val[grad_ind]

    # Add constraint
    for grad_ind, grad in enumerate(gradient):
      lkld_val[grad_ind] += self._GetConstraint(Y, derivative=grad, column_1=column_1, column_2=column_2)

    # Multiply by constant
    lkld_val = [val*multiply_by for val in lkld_val]

    # End output
    end_time = time.time()

    # Calculate minimisation step
    if self.minimisation_step is not None:
      self.minimisation_step += 1
      if self.verbose:
        print(f"Step={self.minimisation_step}")

    # Print Y
    print(f"  Y: ")
    for ind, Y_col in enumerate(self.Y_columns):
      print(f"    {Y_col} = {float(Y.to_numpy().flatten()[ind])}")

    # Print result
    print(f"  Result: ")
    for grad in gradient:
      result = lkld_val[gradient.index(grad)]
      if grad == 0:
        print(f"    {multiply_by}lnL = {result}")
      elif grad == 1:
        for col_ind, col in enumerate(column_1):
          print(f"    {multiply_by}dlnL/d{col} = {result[col_ind]}")
      elif grad == 2:
        if column_1 == column_2:
          print(f"    {multiply_by}d^2lnL/d{column_1[0]}^2 = {result}")
        else:
          print(f"    {multiply_by}d^2lnL/d{column_1[0]}_d{column_2[0]} = {result}")

    # Print time
    print(f"  Time-taken: {round(end_time-start_time,2)} seconds")
      
    # Convert back
    if convert_back:
      lkld_val = lkld_val[0]

    return lkld_val


  def LikelihoodBinned(self, bin_values, Y, gradient=[0], column_1=None, column_2=None, category=None):

    ln_lkld = []
    for grad in gradient:
      if grad == 0:
        ln_lkld += [self._GetBinned(bin_values, Y, category=category)]
      elif grad == 1:
        ln_lkld_func = partial(self._GetBinned, bin_values, category=category)
        ln_lkld += [self._HelperNumericalGradientFromLinear(ln_lkld_func, Y, column_1, None, gradient=1)]
      elif grad == 2:
        if column_1 is None or column_2 is None:
          raise ValueError("You must specify column_1 and column_2 to get the second derivative.")
        else:
          ln_lkld_func = partial(self._GetBinned, bin_values, category=category)
          grad_func = lambda Y, func, column: self._HelperNumericalGradientFromLinear(func, Y, column, None, gradient=1)
          locked_grad_func = partial(grad_func, func=ln_lkld_func, column=column_1)
          ln_lkld += [self._HelperNumericalGradientFromLinear(locked_grad_func, Y, column_2, None, gradient=1)]

    return ln_lkld


  def LikelihoodBinnedExtended(self, bin_values, Y, gradient=[0], column_1=None, column_2=None, category=None):
 
    ln_lkld = []
    for grad in gradient:
      if grad == 0:
        ln_lkld += [self._GetBinnedExtended(bin_values, Y, category=category)]
      elif grad == 1:
        ln_lkld_func = partial(self._GetBinnedExtended, bin_values, category=category)
        ln_lkld += [self._HelperNumericalGradientFromLinear(ln_lkld_func, Y, column_1, None, gradient=1)]
      elif grad == 2:
        if column_1 is None or column_2 is None:
          raise ValueError("You must specify column_1 and column_2 to get the second derivative.")
        else:
          ln_lkld_func = partial(self._GetBinnedExtended, bin_values, Y, category=category)
          grad_func = lambda Y, func, column: self._HelperNumericalGradientFromLinear(func, Y, column, None, gradient=1)
          locked_grad_func = partial(grad_func, func=ln_lkld_func, column=column_1)
          ln_lkld += [self._HelperNumericalGradientFromLinear(locked_grad_func, Y, column_2, None, gradient=1)]

    return ln_lkld


  def LikelihoodUnbinned(self, X_dps, Y, gradient=[0], column_1=None, column_2=None, category=None):
    """
    Computes the likelihood for unbinned data.

    Args:
      X (data processor): The independent variables.
      Y (array): The model parameters.
      gradient (int or list): The gradient to compute (0, 1, or 2).
      column_1 (str or list): The first column for the gradient.
      column_2 (str or list): The second column for the gradient.
      before_sum (bool): If True, returns the full array before summing.

    Returns:
        float: The likelihood value.

    """

    # Get event loop value
    first_loop = True
    self._HelperSetSeed()
    for X_dp in X_dps:
      dps_ln_lkld = X_dp.GetFull(
        method = "custom",
        custom = self._CustomDPMethod,
        custom_options = {"function": self._GetEventLoopUnbinned, "Y": Y, "options": {"wt_name":X_dp.wt_name, "gradient":gradient, "column_1":column_1, "column_2":column_2, "category":category}}
      )
      if first_loop:
        ln_lkld = copy.deepcopy(dps_ln_lkld)
        first_loop = False
      else:
        ln_lkld = [ln_lkld[ind]+dps_ln_lkld[ind] for ind in range(len(ln_lkld))]

    return ln_lkld

  def LikelihoodUnbinnedExtended(self, X_dps, Y, gradient=[0], column_1=None, column_2=None, category=None):

    # Get event loop value
    first_loop = True
    self._HelperSetSeed()
    for X_dp in X_dps:
      dps_ln_lkld = X_dp.GetFull(
        method = "custom",
        custom = self._CustomDPMethod,
        custom_options = {"function": self._GetEventLoopUnbinnedExtended, "Y": Y, "options": {"wt_name":X_dp.wt_name, "gradient":gradient, "column_1":column_1, "column_2":column_2, "category":category}}
      )
      if first_loop:
        ln_lkld = copy.deepcopy(dps_ln_lkld)
        first_loop = False
      else:
        ln_lkld = [ln_lkld[ind]+dps_ln_lkld[ind] for ind in range(len(ln_lkld))]

    # Add poisson term
    for grad_ind, grad in enumerate(gradient):
      for name, _ in self.models["yields"][category].items():
        ln_lkld[grad_ind] -= self._GetYieldGradient(name, Y, gradient=grad, column_1=column_1, column_2=column_2, category=category)

    return ln_lkld


  def GetBestFit(self, X_dps, initial_guess, method="scipy", freeze={}, initial_step_size=0.2, save_best_fit=True):
    """
    Finds the best-fit parameters using numerical optimization.

    Args:
        X (array): The independent variables.
        initial_guess (array): Initial values for the model parameters.
        wts (array): The weights for the data points (optional).

    """

    if method in ["scipy"]:

      func_to_minimise = lambda Y: self.Run(X_dps, Y, multiply_by=-2, gradient=0)
      func, initial_guess = self._HelperFreeze(freeze, initial_guess.to_numpy().flatten(), func_to_minimise)
      result = self.Minimise(func, initial_guess, initial_step_size=initial_step_size, method=method)
      
    elif method in ["minuit"]:

      inputs = ",".join(['p'+str(ind) for ind in range(len(initial_guess.to_numpy().flatten()))])
      func_to_minimise = eval(f"lambda {inputs}: self.Run(X_dps, np.array([{inputs}]), multiply_by=-2, gradient=0)", {"self": self, "X_dps": X_dps, 'np': np})
      func, initial_guess = self._HelperFreeze(freeze, initial_guess.to_numpy().flatten(), func_to_minimise, non_list_input=True)
      result = self.Minimise(func, initial_guess, method=method, initial_step_size=initial_step_size)

    elif method in ["scipy_with_gradients","minuit_with_gradients","custom"]:

      func_val_and_jac = lambda Y: self.Run(X_dps, Y, multiply_by=-2, gradient=[0,1])

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
            _, jac = func_val_and_jac(Y)
            return jac
      nllgrad = NLLAndGradient()

      if method in ["scipy_with_gradients","custom"]:
        func, jac, initial_guess = self._HelperFreeze(freeze, initial_guess.to_numpy().flatten(), nllgrad.GetNLL, jac=nllgrad.GetJac)
        result = self.Minimise(func, initial_guess, initial_step_size=initial_step_size, method=method, jac=jac)
      elif method in ["minuit_with_gradients"]:
        inputs = ",".join(['p'+str(ind) for ind in range(len(initial_guess.to_numpy().flatten()))])
        nll_func = eval(f"lambda {inputs}: nllgrad.GetNLL(np.array([{inputs}]))", {"nllgrad": nllgrad, 'np': np})
        nll_jac = eval(f"lambda {inputs}: nllgrad.GetJac(np.array([{inputs}]))", {"nllgrad": nllgrad, 'np': np})
        func, jac, initial_guess = self._HelperFreeze(freeze, initial_guess.to_numpy().flatten(), nll_func, jac=nll_jac, non_list_input=True)
        result = self.Minimise(func, initial_guess, method=method, jac=jac)

    if save_best_fit:
      self.best_fit = result[0]
      self.best_fit_nll = result[1]

    return result

  def GetCovariance(self):

    hessian = np.array(self.hessian)
    inverse_hessian = np.linalg.inv(hessian)

    if self.D_matrix is None:
      covariance = inverse_hessian
    else:

      hessian_columns = self.hessian_columns if self.hessian_columns is not None else self.Y_columns
      D_matrix_columns_inds = [hessian_columns.index(i) for i in self.D_matrix_columns]
      inverse_hessian_D_matrix_columns = inverse_hessian[:,D_matrix_columns_inds][D_matrix_columns_inds,:]
      covariance_D_matrix_columns = inverse_hessian_D_matrix_columns @ self.D_matrix @ inverse_hessian_D_matrix_columns
      covariance = np.zeros((len(hessian_columns), len(hessian_columns)))
      for ind_1, col_1 in enumerate(hessian_columns):
        for ind_2, col_2 in enumerate(hessian_columns):
          if col_1 in self.D_matrix_columns and col_2 in self.D_matrix_columns:
            covariance[ind_1][ind_2] = covariance_D_matrix_columns[self.D_matrix_columns.index(col_1)][self.D_matrix_columns.index(col_2)]
          else:
            covariance[ind_1][ind_2] = inverse_hessian[ind_1][ind_2]

    return covariance.tolist()


  def GetCovarianceIntervals(self, covariance, column):

    col_index = self.Y_columns.index(column)
    uncertainty = np.sqrt(covariance[col_index][col_index])
    m1p1_vals = {-1: uncertainty, 1: uncertainty}
    return m1p1_vals


  def GetDMatrix(self, X_dps, scan_over):


    # Get the D matrix
    if self.type in ["unbinned","unbinned_extended"]:
      d_matrix = 0
      self._HelperSetSeed()
      for cat in self.categories:
        sum_wts = 0.0
        for X_dp in X_dps[cat]:
          sum_wts += X_dp.GetFull(method="sum")
        for X_dp in X_dps[cat]:
          # Get D matrix
          dps_d_matrix = X_dp.GetFull(
            method = "custom",
            custom = self._CustomDPMethodForDMatrix,
            custom_options = {"Y" : pd.DataFrame([self.best_fit], columns=self.Y_columns), "wt_name" : X_dp.wt_name, "scan_over" : scan_over, "sum_wts" : sum_wts, "category" : cat}
          )
          d_matrix += dps_d_matrix
    else:
      raise ValueError("D matrix only valid for unbinned fits.")

    # Remove nan columns or rows
    non_nan_cols = [scan_over[ind] for ind, row in enumerate(d_matrix) if not np.isnan(row).all()]
    non_nan_d_matrix = np.zeros((len(non_nan_cols), len(non_nan_cols)))
    for ind_1, col_1 in enumerate(non_nan_cols):
      for ind_2, col_2 in enumerate(non_nan_cols):
        non_nan_d_matrix[ind_1][ind_2] = d_matrix[scan_over.index(col_1)][scan_over.index(col_2)]

    return non_nan_d_matrix.tolist(), non_nan_cols


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
        nll = self.Run(X_dps, Y, multiply_by=-2) - self.best_fit_nll

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
        nll = self.Run(X_dps, Y, multiply_by=-2) - self.best_fit_nll
        est_sig = np.sqrt(nll)
        if nll > 0.0:
          m1p1_vals[sign] = m1p1_vals[sign]/est_sig

      if fin: 
        nll_could_be_neg = False

    if self.verbose:
      print(f"Estimated result: {self.best_fit[col_index]} + {m1p1_vals[1]} - {m1p1_vals[-1]}")

    return m1p1_vals


  def GetScanXValues(self, X_dps, column, estimated_sigmas_shown=3.2, estimated_sigma_step=0.4, initial_step_fraction=0.001, min_step=0.1, method="approximate"):
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
    if method == "approximate":
      m1p1_vals = self.GetApproximateUncertainty(X_dps, column, initial_step_fraction=initial_step_fraction, min_step=min_step)
    elif method == "hessian":
      m1p1_vals = self.GetCovarianceIntervals(self.GetCovariance(), column)

    if self.verbose:
      print(f"Estimated result: {float(self.best_fit[col_index])} + {m1p1_vals[1]} - {m1p1_vals[-1]}")

    lower_scan_vals = [float(self.best_fit[col_index] - estimated_sigma_step*ind*m1p1_vals[-1]) for ind in range(int(np.ceil(estimated_sigmas_shown/estimated_sigma_step)),0,-1)]
    upper_scan_vals = [float(self.best_fit[col_index] + estimated_sigma_step*ind*m1p1_vals[1]) for ind in range(1,int(np.ceil(estimated_sigmas_shown/estimated_sigma_step))+1)]

    return lower_scan_vals + [float(self.best_fit[col_index])] + upper_scan_vals

  def Minimise(self, func, initial_guess, method="scipy", initial_step_size=0.02, jac=None):
    """
    Minimizes the given function using numerical optimization.

    Args:
        func (callable): The objective function to be minimized.
        initial_guess (array): Initial values for the parameters.

    Returns:
        tuple: Best-fit parameters and the corresponding function value.

    """

    # Set minimisation_step
    self.minimisation_step = 0

    # scipy
    if method == "scipy":
      # get initial simplex
      n = len(initial_guess)
      initial_simplex = [initial_guess]
      for i in range(n):
        row = copy.deepcopy(initial_guess)
        row[i] += initial_step_size * max(row[i], 1.0)
        initial_simplex.append(row)
      # run minimisation
      minimisation = minimize(func, initial_guess, method='Nelder-Mead', tol=0.001, options={'xatol': 0.001, 'fatol': 0.01, 'initial_simplex': initial_simplex})
      res = minimisation.x, minimisation.fun
    
    # scipy with gradients
    elif method == "scipy_with_gradients":
      minimisation = minimize(func, initial_guess, jac=jac, method='L-BFGS-B', tol=0.001, options={'ftol': 1e-9, 'gtol':1e-8})
      res = minimisation.x, minimisation.fun

    # minuit
    elif method == "minuit":
      minuit_initial_guess = {f"p{ind}":val for ind, val in enumerate(initial_guess)}
      m = Minuit(func, **minuit_initial_guess)
      for params in range(len(initial_guess)):
        m.errors[f"p{params}"] *= 1
      m.errordef = Minuit.LIKELIHOOD
      m.strategy = 2
      m.tol = 0.01
      m.simplex()
      m.migrad()
      res = m.values, m.fval

    # minuit with gradients
    elif method == "minuit_with_gradients":
      minuit_initial_guess = {f"p{ind}":val for ind, val in enumerate(initial_guess)}
      inputs = ",".join(['p'+str(ind) for ind in range(len(initial_guess))])
      class_code = f"""
class NLLAndGradient():
    def __init__(self, func, jac):
        self.func = func
        self.jac = jac
    def __call__(self, {inputs}):
        return self.func({inputs})
    def grad(self,{inputs}):
        return self.jac({inputs})
      """
      namespace = {}
      exec(class_code, globals(), namespace)
      NLLAndGradient = namespace["NLLAndGradient"]
      cost = NLLAndGradient(func, jac)

      m = Minuit(cost, **minuit_initial_guess)
      for params in range(len(initial_guess)):
        m.errors[f"p{params}"] *= 10
      m.errordef = Minuit.LIKELIHOOD
      m.strategy = 0
      m.tol = 0.01
      m.migrad()
      res = m.values, m.fval

    # custom
    elif method == "custom":     
      res = CustomMinimise(func, jac, initial_guess)

    else:
      raise ValueError(f"Method {method} not recognised. Please use 'scipy', 'scipy_with_gradients', 'minuit', 'minuit_with_gradients' or 'custom'.")

    # Set minimisation_step
    self.minimisation_step = None

    return res


  def GetAndWriteApproximateUncertaintyToYaml(self, X_dps, col, row=None, filename="approx_crossings.yaml", symmetrise=True):  

    uncerts = self.GetApproximateUncertainty(X_dps, col)

    if symmetrise:
      dev = (uncerts[1] + uncerts[-1])/2
      uncerts = {
        -1 : dev,
        1 : dev
      }

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

  def GetAndWriteCovarianceToYaml(self, row=None, scan_over=None, filename="covariance.yaml"):

    self.covariance = self.GetCovariance()
    
    dump = {
      "columns": self.Y_columns, 
      "matrix_columns" : scan_over, 
      "covariance" : self.covariance,
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
      "best_fit": [float(i) for i in self.best_fit], 
    }
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)


  def GetAndWriteCovarianceIntervalsToYaml(self, col, row=None, scan_over=None, filename="covariance_results.yaml"):
    
    m1p1_vals = self.GetCovarianceIntervals(self.covariance, col)
    col_index = self.Y_columns.index(col)
    best_fit_col = self.best_fit[col_index]
    dump = {
      "columns": self.Y_columns, 
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
      "varied_column" : col,
      "crossings" : {
        -2 : float(best_fit_col-2*m1p1_vals[-1]),
        -1 : float(best_fit_col-m1p1_vals[-1]),
        0 : float(best_fit_col),
        1 : float(best_fit_col+m1p1_vals[1]),
        2 : float(best_fit_col+2*m1p1_vals[1]),
      }
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

    D_matrix, scan_over = self.GetDMatrix(X_dps, scan_over)

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


  def GetAndWriteHessianToYaml(self, X_dps, row=None, freeze={}, scan_over=None, filename="hessian.yaml", specific_column_1=None, specific_column_2=None, numerical=False):
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
    if numerical:

      minuit_func_input = ",".join(['p'+str(ind) for ind in range(len(scan_over))])
      pd_func_input = ",".join([minuit_func_input] + list(freeze.values()))
      pd_func_cols = ",".join([f'"{k}"' for k in scan_over + list(freeze.keys())])
      run_func_input = f"pd.DataFrame([[{pd_func_input}]], columns=[{pd_func_cols}])"
      best_fit_input = ",".join([f"p{ind}={self.best_fit[ind]}" for ind in range(len(self.Y_columns))])
      minuit_func = eval(f"lambda {minuit_func_input}: self.Run(X_dps, {run_func_input}, multiply_by=-2, gradient=0)", {"self": self, "X_dps": X_dps, 'pd': pd})
      minuit_class = eval(f"Minuit(minuit_func, {best_fit_input})", {"Minuit": Minuit, "minuit_func": minuit_func, "self": self, 'pd': pd})
      for params in range(len(self.Y_columns)):
        minuit_class.values[f"p{params}"] = self.best_fit[params]
      minuit_class.hesse()
      cov = minuit_class.covariance
      hessian = np.linalg.inv(cov)

    else:

      hessian = np.zeros((len(scan_over),len(scan_over)))
      for index_1, column_1 in enumerate(scan_over):
        for index_2, column_2 in enumerate(scan_over):
          if index_1 <= index_2:
            if not (specific_column_1 is None or specific_column_2 is None):
              if not(column_1 == specific_column_1 and column_2 == specific_column_2):
                continue
            hessian[index_1, index_2] = self.Run(X_dps, self.best_fit, multiply_by=-1, gradient=2, column_1=column_1, column_2=column_2)

      # Make it into a full square
      for index_1, column_1 in enumerate(scan_over):
        for index_2, column_2 in enumerate(scan_over):
          if index_1 > index_2:
            hessian[index_1, index_2] = hessian[index_2, index_1]

    # Make into list
    hessian = hessian.tolist()

    # Dump
    dump = {
      "columns": self.Y_columns, 
      "matrix_columns" : scan_over, 
      "hessian" : hessian,
      "row" : [float(row.loc[0,col]) for col in self.Y_columns] if row is not None else None,
      "best_fit": [float(i) for i in self.best_fit], 
    }
    if self.verbose:
      pprint(dump)
    print(f"Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)


  def GetAndWriteScanRangesToYaml(self, X_dps, col, row=None, filename="scan_ranges.yaml", estimated_sigmas_shown=3.2, estimated_sigma_step=0.4, method="approximate"):
    """
    Computes the scan ranges for a given column and writes them to a YAML file.

    Args:
        X (array): The independent variables.
        row (array): The row values.
        col (str): The column name for the parameter to be scanned.
        wt (array): The weights for the data points (optional).
        filename (str): The name of the YAML file (default is "scan_ranges.yaml").
    """

    scan_values = self.GetScanXValues(X_dps, col, estimated_sigmas_shown=estimated_sigmas_shown, estimated_sigma_step=estimated_sigma_step, method=method)

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
      result = self.Run(X_dps, Y, multiply_by=-2)
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