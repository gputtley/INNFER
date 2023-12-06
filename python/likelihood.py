import copy
import yaml
import time
import numpy as np
from scipy.optimize import minimize

class Likelihood():
  """
  A class representing a likelihood function.
  """
  def __init__(self, models, type="unbinned", parameters={}, data_parameters={}):
    """
    Initializes a Likelihood object.

    Args:
        models (dict): A dictionary containing model names as keys and corresponding probability density functions (PDFs) as values.
        type (str): Type of likelihood, either "unbinned" or "unbinned_extended" (default is "unbinned").
        parameters (dict): A dictionary containing model parameters (default is an empty dictionary).
        data_parameters (dict): A dictionary containing data parameters (default is an empty dictionary).
    """
    self.type = type # unbinned, unbinned_extended
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

    return Y_columns


  def Run(self, X, Y, wts=None, return_ln=False):
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
      lkld_val = self.Unbinned(X, Y, wts=wts, return_ln=return_ln)
    elif self.type == "unbinned_extended":
      lkld_val = self.UnbinnedExtended(X, Y, wts=wts, return_ln=return_ln)

    return lkld_val

  def _LogSumExpTrick(self, ln_a, ln_b):
    mask = ln_b > ln_a
    ln_a[mask], ln_b[mask] = ln_b[mask], ln_a[mask]
    return ln_a + np.log1p(np.exp(ln_b - ln_a))

  def Unbinned(self, X, Y, wts=None, return_ln=False):
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

    Y = list(Y)

    first_loop = True
    sum_rate_params = 0.0
    for name, pdf in self.models["pdfs"].items():

      if "mu_"+name in self.Y_columns:
        rate_param = Y[self.Y_columns.index("mu_"+name)]
      else:
        rate_param = 1.0
      if rate_param == 0.0: continue
      sum_rate_params += rate_param

      # Get rate times probability
      log_p = pdf.Probability(copy.deepcopy(X), np.array([Y]), y_columns=self.Y_columns, return_log_prob=True)
      ln_lklds_with_rate_params = np.log(rate_param) + log_p

      # Sum together probabilities of different files
      if first_loop:
        ln_lklds = copy.deepcopy(ln_lklds_with_rate_params)
        first_loop = False
      else:
        ln_lklds = self._LogSumExpTrick(ln_lklds,ln_lklds_with_rate_params)

    # Rescale so total pdf integrate to 1
    ln_lklds -= np.log(sum_rate_params)

    # Weight the events
    if wts is not None:
      ln_lklds = wts.flatten()*ln_lklds

    # Product the events
    ln_lkld = ln_lklds.sum()

    # Add constraints
    if "nuisance_constraints" in self.parameters.keys():
      for k, v in self.parameters["nuisance_constraints"].items():
        if v == "Gaussian":
          constraint = self._Gaussian(Y[self.Y_columns.index(k)])
        elif v == "LogNormal":
          constraint = self._LogNormal(Y[self.Y_columns.index(k)])
        ln_lkld += np.log(constraint)

    print(f">> Y={Y}, lnL={ln_lkld}")
    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

  def UnbinnedExtended(self, X, Y, wts=None, return_ln=False):
    """
    Computes the extended likelihood for unbinned data.

    Args:
        X (array): The independent variables.
        Y (array): The model parameters.
        wts (array): The weights for the data points (optional).
        return_ln (bool): Whether to return the natural logarithm of the likelihood (optional).

    Returns:
        float: The likelihood value.

    """
    Y = list(Y)

    first_loop = True
    sum_rate_params = 0.0
    for name, pdf in self.models["pdfs"].items():

      if "mu_"+name in self.Y_columns:
        rate_param = Y[self.Y_columns.index("mu_"+name)]
      else:
        rate_param = 1.0
      if rate_param == 0.0: continue
      sum_rate_params += rate_param

      # Get rate times probability
      log_p = pdf.Probability(copy.deepcopy(X), np.array([Y]), y_columns=self.Y_columns, return_log_prob=True)
      # NEED TO PUT YIELD SCALING FUNCTIONS HERE
      ln_lklds_with_rate_params = np.log(rate_param) + log_p

      # Sum together probabilities of different files
      if first_loop:
        ln_lklds = copy.deepcopy(ln_lklds_with_rate_params)
        first_loop = False
      else:
        ln_lklds = self._LogSumExpTrick(ln_lklds,ln_lklds_with_rate_params)

    # NEED TO PUT NEGATIVE EXPONENTIAL OF YIELD SCALING FUNCTIONS

    # Weight the events
    if wts is not None:
      ln_lklds = wts.flatten()*ln_lklds

    # Product the events
    ln_lkld = ln_lklds.sum()

    # Add constraints
    if "nuisance_constraints" in self.parameters.keys():
      for k, v in self.parameters["nuisance_constraints"].items():
        if v == "Gaussian":
          constraint = self._Gaussian(Y[self.Y_columns.index(k)])
        elif v == "LogNormal":
          constraint = self._LogNormal(Y[self.Y_columns.index(k)])
        ln_lkld += np.log(constraint)

    print(f">> Y={Y}, lnL={ln_lkld}")
    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)


  def _Gaussian(self, x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

  def _LogNormal(self, x):
    return 1 / (x * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (np.log(x))**2)

  def GetBestFit(self, X, initial_guess, wts=None):
    """
    Finds the best-fit parameters using numerical optimization.

    Args:
        X (array): The independent variables.
        initial_guess (array): Initial values for the model parameters.
        wts (array): The weights for the data points (optional).

    """
    def NLL(Y): 
      nll = -2*self.Run(X, Y, wts=wts, return_ln=True)
      #print(Y, nll)
      return nll
    result = self.Minimise(NLL, initial_guess)
    self.best_fit = result[0]
    self.best_fit_nll = result[1]

  def GetScanXValues(self, X, column, wts=None, estimated_sigmas_shown=5, estimated_sigma_step=0.2, initial_step_fraction=0.001, min_step=0.1):

    col_index = self.Y_columns.index(column)
    nll_could_be_neg = True
    while nll_could_be_neg:
      m1p1_vals = {}
      fin = True
      for sign in [1,-1]:
        Y = copy.deepcopy(self.best_fit)
        step = max(self.best_fit[col_index]*initial_step_fraction,min_step)
        Y[col_index] = self.best_fit[col_index] + sign*step
        nll = -2*self.Run(X, Y, wts=wts, return_ln=True) - self.best_fit_nll
        if nll < 0.0:
          self.best_fit[col_index] = Y[col_index]
          self.best_fit_nll += nll
          fin = False
          break
        est_sig = np.sqrt(nll)
        m1p1_vals[sign] = step/est_sig
      if fin: nll_could_be_neg = False

    lower_scan_vals = [float(self.best_fit[col_index] - estimated_sigma_step*ind*m1p1_vals[-1]) for ind in range(int(np.ceil(estimated_sigmas_shown/estimated_sigma_step)),0,-1)]
    upper_scan_vals = [float(self.best_fit[col_index] + estimated_sigma_step*ind*m1p1_vals[1]) for ind in range(1,int(np.ceil(estimated_sigmas_shown/estimated_sigma_step))+1)]

    return lower_scan_vals + [float(self.best_fit[col_index])] + upper_scan_vals

  def GetAndWriteBestFitToYaml(self, X, row, initial_guess, wt=None, filename="best_fit.yaml"):

    self.GetBestFit(X, np.array(initial_guess), wts=wt)
    dump = {
      "row" : [float(i) for i in row],
      "columns": self.Y_columns, 
      "best_fit": [float(i) for i in self.best_fit], 
      "best_fit_nll": float(self.best_fit_nll)
      }
    print(dump)
    print(f">> Created {filename}")
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)

  def GetAndWriteScanRangesToYaml(self, X, row, col, wt=None, filename="scan_ranges.yaml"):

    scan_values = self.GetScanXValues(X, col, wts=wt)
    dump = {
      "row" : [float(i) for i in row],
      "columns" : self.Y_columns, 
      "varied_column" : col,
      "scan_values" : scan_values
    }
    print(dump)
    print(f">> Created {filename}")
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)

  def GetAndWriteNLLToYaml(self, X, row, col, col_val, wt=None, filename="nlls.yaml"):
    
    col_index = self.Y_columns.index(col)
    Y = copy.deepcopy(self.best_fit)
    Y[col_index] = col_val
    nll = -2*self.Run(X, Y, wts=wt, return_ln=True) - self.best_fit_nll
    dump = {
      "row" : [float(i) for i in row],
      "columns" : self.Y_columns, 
      "varied_column" : col,
      "nlls": [float(nll)], 
      "scan_values" : [float(col_val)],
    }
    print(f">> Created {filename}")
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)  


  def MakeScanInSeries(self, X, column, wts=None, estimated_sigmas_shown=3, estimated_sigma_step=0.2, initial_step_fraction=0.001):
    """
    Performs a parameter scan for a given column.

    Args:
        X (array): The independent variables.
        column (str): The column name for the parameter to be scanned.
        wts (array): The weights for the data points (optional).
        estimated_sigmas_shown (int): Number of estimated sigmas to be shown in the scan (default is 3).
        estimated_sigma_step (float): Step size for estimated sigmas in the scan (default is 0.2).
        initial_step_fraction (float): Fraction of the initial value to use as the step size (default is 0.001).

    Returns:
        tuple: Lists of X values, likelihood values, and crossing points.

    """
    col_index = self.Y_columns.index(column)
    x_scan = {1:[],-1:[]}
    y_scan = {1:[],-1:[]}
    for sign in [1,-1]:
      nll_could_be_neg = True
      while nll_could_be_neg:
        fin = True
        x_scan[sign] = [self.best_fit[col_index]]
        y_scan[sign] = [0.0]
        Y = copy.deepcopy(self.best_fit)
        #print("Sign", sign)
        # initial guess
        step = self.best_fit[col_index]*initial_step_fraction
        Y[col_index] = self.best_fit[col_index] + sign*step
        nll = -2*self.Run(X, Y, wts=wts, return_ln=True) - self.best_fit_nll
        if nll < 0.0:
          self.best_fit[col_index] = Y[col_index]
          self.best_fit_nll += nll
          x_scan[sign].append(Y[col_index])
          y_scan[sign].append(nll)
          for k, v in y_scan.items(): y_scan[k] = [x - nll for x in v]
          fin = False
          break
        #print("Initial", Y[col_index], nll, step)
        est_sig = np.sqrt(nll)
        if est_sig <= estimated_sigmas_shown:
          x_scan[sign].append(Y[col_index])
          y_scan[sign].append(nll)
        for ind in range(int(np.ceil(estimated_sigmas_shown/estimated_sigma_step))):
          # recalculate estimated unit step size
          unit_step = step/est_sig
          step = estimated_sigma_step*(ind+1)*unit_step
          # Get next requested shift value
          Y[col_index] = self.best_fit[col_index] + sign*step
          nll = -2*self.Run(X, Y, wts=wts, return_ln=True) - self.best_fit_nll
          if nll < 0.0:
            self.best_fit[col_index] = Y[col_index]
            self.best_fit_nll += nll
            x_scan[sign].append(Y[col_index])
            y_scan[sign].append(nll)
            for k, v in y_scan.items(): y_scan[k] = [x - nll for x in v]
            fin = False
            break
          est_sig = np.sqrt(nll) 
          #print("Other", Y[col_index], nll, step, unit_step, est_sig)
          # add to store
          x_scan[sign].append(Y[col_index])
          y_scan[sign].append(nll)
        if fin: nll_could_be_neg = False
    x = x_scan[-1] + x_scan[1]
    y = y_scan[-1] + y_scan[1]
    sorted_lists = sorted(zip(x, y))
    x, y = zip(*sorted_lists)
    crossings = self.FindCrossings(list(x), list(y), self.best_fit[col_index], crossings=range(estimated_sigmas_shown))
    return list(x), list(y), crossings

  def FindCrossings(self, x, y, crossings=[1, 2]):
    """
    Finds the intersection points for the likelihood curve.

    Args:
        x (list): List of X values.
        y (list): List of likelihood values.
        best_fit (float): The best-fit value for the parameter.
        crossings (list): List of crossing points to find (default is [1, 2]).

    Returns:
        dict: Dictionary containing crossing points and their corresponding values.

    """
    values = {}
    values[0] = float(x[y.index(min(y))])
    for crossing in crossings:
      for sign in [-1, 1]:
        condition = np.array(x) * sign > values[0] * sign
        filtered_x = np.array(x)[condition]
        filtered_y = np.array(y)[condition]
        sorted_indices = np.argsort(filtered_y)
        filtered_x = filtered_x[sorted_indices]
        filtered_y = filtered_y[sorted_indices]
        if crossing**2 > min(filtered_y) and crossing**2 < max(filtered_y):
          values[sign * crossing] = float(np.interp(crossing**2, filtered_y, filtered_x))
    return values

  def Minimise(self, func, initial_guess):
    """
    Minimizes the given function using numerical optimization.

    Args:
        func (callable): The objective function to be minimized.
        initial_guess (array): Initial values for the parameters.

    Returns:
        tuple: Best-fit parameters and the corresponding function value.

    """
    minimisation = minimize(func, initial_guess, method='Nelder-Mead')
    #minimisation = minimize(func, initial_guess, method='Nelder-Mead', tol=0.01, options={'xatol': 0.001, 'fatol': 0.01, 'maxiter': 20})
    return minimisation.x, minimisation.fun