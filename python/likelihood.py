import copy
import yaml
import time
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from plotting import plot_likelihood, plot_histograms
from other_functions import GetYName, MakeDirectories
from pprint import pprint

class Likelihood():
  """
  A class representing a likelihood function.
  """
  def __init__(self, models, type="unbinned_extended", parameters={}, data_parameters={}):
    """
    Initializes a Likelihood object.

    Args:
        models (dict): A dictionary containing model names as keys and corresponding probability density functions (PDFs) as values.
        type (str): Type of likelihood, either "unbinned" or "unbinned_extended" (default is "unbinned").
        parameters (dict): A dictionary containing model parameters (default is an empty dictionary).
        data_parameters (dict): A dictionary containing data parameters (default is an empty dictionary).
    """
    self.type = type # unbinned, unbinned_extended, binned_extended
    self.models = models
    self.parameters = parameters
    self.data_parameters = data_parameters
    self.Y_columns = self._MakeY()
    self.debug_mode = False
    self.debug_hists = {}
    self.debug_bins = {}
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


  def Run(self, X, Y, wts=None, return_ln=False, before_sum=False):
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
      lkld_val = self.Unbinned(X, Y, wts=wts, return_ln=return_ln, before_sum=before_sum)
    elif self.type == "unbinned_extended":
      lkld_val = self.UnbinnedExtended(X, Y, wts=wts, return_ln=return_ln, before_sum=before_sum)
    elif self.type == "binned_extended":
      lkld_val = self.BinnedExtended(X, Y, wts=wts, return_ln=return_ln)

    return lkld_val

  def _LogSumExpTrick(self, ln_a, ln_b):
    mask = ln_b > ln_a
    ln_a[mask], ln_b[mask] = ln_b[mask], ln_a[mask]
    return ln_a + np.log1p(np.exp(ln_b - ln_a))

  def _GetYield(self, file_name, Y, y_columns=None):
    Y = np.array(Y)
    if y_columns is not None:
      column_indices = [y_columns.index(col) for col in self.data_parameters[file_name]["Y_columns"]]
      Y = Y[column_indices]
    yd = self.models["yields"][file_name](Y)
    return yd

  def Unbinned(self, X, Y, wts=None, return_ln=False, before_sum=False, verbose=True):
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

      if self.debug_mode:
        print(">> Making debug histograms")
        for x_col in range(X.shape[1]):
          y_name = GetYName(Y, purpose='plot', prefix="y=")
          col_name = self.data_parameters[name]['X_columns'][x_col]
          hist, bins = np.histogram(X[:,x_col], weights=wts.flatten()*log_p.flatten(), bins=40)
          if col_name not in self.debug_hists.keys():
            self.debug_hists[col_name] = {}
          if y_name not in self.debug_hists[col_name].keys():
            self.debug_hists[col_name][y_name] = {}
          self.debug_hists[col_name][y_name] = copy.deepcopy(hist)
          if col_name not in self.debug_bins.keys():
            self.debug_bins[col_name] = bins

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

    if before_sum:
      return ln_lklds

    # Product the events
    ln_lkld = np.sum(ln_lklds, dtype=np.float128)

    # Add constraints
    if "nuisance_constraints" in self.parameters.keys():
      for k, v in self.parameters["nuisance_constraints"].items():
        if v == "Gaussian":
          constraint = self._Gaussian(Y[self.Y_columns.index(k)])
        elif v == "LogNormal":
          constraint = self._LogNormal(Y[self.Y_columns.index(k)])
        ln_lkld += np.log(constraint)

    if verbose:
      print(f">> Y={Y}, lnL={ln_lkld}")

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

  def UnbinnedExtended(self, X, Y, wts=None, return_ln=False, before_sum=False, verbose=True):
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
    start_time = time.time()
    Y = list(Y)

    first_loop = True
    for name, pdf in self.models["pdfs"].items():

      if "mu_"+name in self.Y_columns:
        rate_param = Y[self.Y_columns.index("mu_"+name)]
      else:
        rate_param = 1.0
      if rate_param == 0.0: continue

      # Get rate times yield times probability
      log_p = pdf.Probability(copy.deepcopy(X), np.array([Y]), y_columns=self.Y_columns, return_log_prob=True)
      ln_lklds_with_rate_params = np.log(rate_param) + np.log(self._GetYield(name, Y, y_columns=self.Y_columns)) + log_p

      # Sum together probabilities of different files
      if first_loop:
        ln_lklds = copy.deepcopy(ln_lklds_with_rate_params)
        first_loop = False
      else:
        ln_lklds = self._LogSumExpTrick(ln_lklds,ln_lklds_with_rate_params)

    # Weight the events
    if wts is not None:
      ln_lklds = wts.flatten()*ln_lklds

    if before_sum:
      return ln_lklds

    # Product the events
    ln_lkld = np.sum(ln_lklds, dtype=np.float128)

    # Add poisson term
    total_rate = 0.0
    for name, pdf in self.models["yields"].items():

      if "mu_"+name in self.Y_columns:
        rate_param = Y[self.Y_columns.index("mu_"+name)]
      else:
        rate_param = 1.0
      if rate_param == 0.0: continue

      total_rate += (rate_param * self._GetYield(name, Y, y_columns=self.Y_columns))
      
    ln_lkld -= total_rate

    # Add constraints
    if "nuisance_constraints" in self.parameters.keys():
      for k, v in self.parameters["nuisance_constraints"].items():
        if v == "Gaussian":
          constraint = self._Gaussian(Y[self.Y_columns.index(k)])
        elif v == "LogNormal":
          constraint = self._LogNormal(Y[self.Y_columns.index(k)])
        ln_lkld += np.log(constraint)

    end_time = time.time()

    if verbose:
      print(f">> Y={Y}, lnL={ln_lkld}, time={round(end_time-start_time,2)}")

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

  def BinnedExtended(self, X, Y, wts=None, return_ln=False, verbose=True):
    """
    Computes the extended likelihood for binned data.

    Args:
        X (array): The independent variables.
        Y (array): The model parameters.
        wts (array): The weights for the data points (optional).
        return_ln (bool): Whether to return the natural logarithm of the likelihood (optional).

    Returns:
        float: The likelihood value.

    """
    start_time = time.time()
    Y = list(Y)

    # Make data histogram
    data_hist, _ = np.histogram(X, weights=wts, bins=self.models["bin_edges"])
    data_hist = data_hist.astype(np.float64)
    ln_lkld = 0.0

    for bin_number in range(len(self.models["bin_edges"])-1):

      yield_sum = 0.0

      for name, yields in self.models["bin_yields"].items():

        if "mu_"+name in self.Y_columns:
          rate_param = Y[self.Y_columns.index("mu_"+name)]
        else:
          rate_param = 1.0
        if rate_param == 0.0: continue

        # Get rate times yield times probability
        Y = np.array(Y)
        if self.Y_columns is not None:
          column_indices = [self.Y_columns.index(col) for col in self.data_parameters[name]["Y_columns"]]
          Y_for_yield = Y[column_indices]
        yield_sum += (yields[bin_number](Y_for_yield) * rate_param)

      #print(bin_number, yield_sum, data_hist[bin_number])
      ln_lkld += ((data_hist[bin_number]*np.log(yield_sum)) - yield_sum)

    # Add constraints
    if "nuisance_constraints" in self.parameters.keys():
      for k, v in self.parameters["nuisance_constraints"].items():
        if v == "Gaussian":
          constraint = self._Gaussian(Y[self.Y_columns.index(k)])
        elif v == "LogNormal":
          constraint = self._LogNormal(Y[self.Y_columns.index(k)])
        ln_lkld += np.log(constraint)

    end_time = time.time()

    if verbose:
      print(f">> Y={Y}, lnL={ln_lkld}, time={round(end_time-start_time,2)}")

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

      
  def _Gaussian(self, x):
    """
    Computes the Gaussian distribution.

    Args:
        x (float): The input value.

    Returns:
        float: The Gaussian distribution value.
    """
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

  def _LogNormal(self, x):
    """
    Computes the Log-Normal distribution.

    Args:
        x (float): The input value.

    Returns:
        float: The Log-Normal distribution value.
    """
    return 1 / (x * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (np.log(x))**2)

  def GetBestFit(self, X, initial_guess, wts=None, method="low_stat_high_stat", freeze={}, initial_step_size=0.05):
    """
    Finds the best-fit parameters using numerical optimization.

    Args:
        X (array): The independent variables.
        initial_guess (array): Initial values for the model parameters.
        wts (array): The weights for the data points (optional).

    """
    if method == "nominal":

      def NLL(Y): 
        nll = -2*self.Run(X, Y, wts=wts, return_ln=True)
        return nll
      result = self.Minimise(NLL, initial_guess, freeze=freeze, initial_step_size=initial_step_size)

    elif method == "increasing_stats":

      class NLL:
        def __init__(self, lkld, full_X, wts=None, start_stats=100, steps_till_full=40):
          self.lkld = lkld
          self.full_X = full_X
          self.full_wts = wts
          self.start_stats = start_stats
          self.steps_till_full = steps_till_full
          self.sum_wts = np.sum(self.full_wts, dtype=np.float128)
          self.iteration = 0

        def objective(self,Y):
          rng = np.random.RandomState(seed=42)
          random_indices = rng.choice(self.full_X.shape[0], int(self.start_stats + self.iteration*((len(self.full_X) - self.start_stats)/self.steps_till_full)), replace=False)
          X = self.full_X[random_indices,:]
          wts = self.full_wts[random_indices,:]
          new_sum_wt = np.sum(wts, dtype=np.float128)
          wts *= self.sum_wts/new_sum_wt
          if self.iteration != self.steps_till_full:
            self.iteration += 1
          nll = -2*self.lkld.Run(X, Y, wts=wts, return_ln=True)
          return nll
               
      nll = NLL(self, X, wts=wts)
      result = self.Minimise(nll.objective, initial_guess, freeze=freeze)

    elif method == "low_stat_high_stat":

      def NLL(Y): 
        nll = -2*self.Run(X, Y, wts=wts, return_ln=True)
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

  def GetScanXValues(self, X, column, wts=None, estimated_sigmas_shown=3.2, estimated_sigma_step=0.4, initial_step_fraction=0.001, min_step=0.1):
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
    """
    Finds the best-fit parameters and writes them to a YAML file.

    Args:
        X (array): The independent variables.
        row (array): The row values.
        initial_guess (array): Initial values for the model parameters.
        wt (array): The weights for the data points (optional).
        filename (str): The name of the YAML file (default is "best_fit.yaml").
    """
    result = self.GetBestFit(X, np.array(initial_guess), wts=wt)
    self.best_fit = result[0]
    self.best_fit_nll = result[1]
    dump = {
      "row" : [float(i) for i in row],
      "columns": self.Y_columns, 
      "best_fit": [float(i) for i in self.best_fit], 
      "best_fit_nll": float(self.best_fit_nll)
      }
    pprint(dump)
    print(f">> Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)

  def GetAndWriteScanRangesToYaml(self, X, row, col, wt=None, filename="scan_ranges.yaml"):
    """
    Computes the scan ranges for a given column and writes them to a YAML file.

    Args:
        X (array): The independent variables.
        row (array): The row values.
        col (str): The column name for the parameter to be scanned.
        wt (array): The weights for the data points (optional).
        filename (str): The name of the YAML file (default is "scan_ranges.yaml").
    """
    scan_values = self.GetScanXValues(X, col, wts=wt)
    dump = {
      "row" : [float(i) for i in row],
      "columns" : self.Y_columns, 
      "varied_column" : col,
      "scan_values" : scan_values
    }
    pprint(dump)
    print(f">> Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)

  def GetAndWriteScanToYaml(self, X, row, col, col_val, wt=None, filename="scan.yaml"):

    col_index = self.Y_columns.index(col)
    Y = copy.deepcopy(self.best_fit)
    Y[col_index] = col_val
    if len(self.Y_columns) > 1:
      print(f">> Profiled fit for {col}={col_val}")
      freeze = {col : col_val}
      result = self.GetBestFit(X, Y, wts=wt, method="nominal", freeze=freeze, initial_step_size=0.001)
      dump = {
        "row" : [float(i) for i in row],
        "columns" : self.Y_columns, 
        "varied_column" : col,
        "nlls": [float(result[1] - self.best_fit_nll)],
        "profiled_columns" : [k for k in self.Y_columns if k != col],
        "profiled_values" : [float(i) for i in result[0]],
        "scan_values" : [float(col_val)],
      }
    else:
      result = -2*self.Run(X, Y, wts=wt, return_ln=True)
      dump = {
        "row" : [float(i) for i in row],
        "columns" : self.Y_columns, 
        "varied_column" : col,
        "nlls": [float(result - self.best_fit_nll)],
        "scan_values" : [float(col_val)],
      }

    pprint(dump)
    print(f">> Created {filename}")
    MakeDirectories(filename)
    with open(filename, 'w') as yaml_file:
      yaml.dump(dump, yaml_file, default_flow_style=False)  

  def GetAndWriteNLLToYaml(self, X, row, col, col_val, wt=None, filename="nlls.yaml"):
    """
    Computes the negative log likelihood (NLL) for a given column value and writes it to a YAML file.

    Args:
        X (array): The independent variables.
        row (array): The row values.
        col (str): The column name for the parameter.
        col_val (float): The column value.
        wt (array): The weights for the data points (optional).
        filename (str): The name of the YAML file (default is "nlls.yaml").
    """    
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
    print(dump)
    print(f">> Created {filename}")
    MakeDirectories(filename)
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
        filtered_x = filtered_x.astype(np.float64)
        filtered_y = filtered_y.astype(np.float64)
        if len(filtered_y) > 1:
          if crossing**2 > min(filtered_y) and crossing**2 < max(filtered_y):
            values[sign * crossing] = float(np.interp(crossing**2, filtered_y, filtered_x))
    return values

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
      minimisation = minimize(func, initial_guess, method='Nelder-Mead', tol=0.01, options={'xatol': 0.001, 'fatol': 0.01, 'initial_simplex': initial_simplex})
      return minimisation.x, minimisation.fun
    
    elif method == "low_stat_high_stat":

      print(">> Doing low stat minimisation first.")

      low_stat_minimisation = minimize(func_low_stat, initial_guess, method='Nelder-Mead', tol=1.0, options={'xatol': 0.001, 'fatol': 1.0, 'initial_simplex': initial_simplex})
      if "pdfs" in self.models.keys():
        for k in self.models["pdfs"].keys():
          self.models["pdfs"][k].probability_store = {}

      print(">> Doing high stat minimisation.")

      # re get initial simplex
      n = len(low_stat_minimisation.x)
      initial_simplex = [low_stat_minimisation.x]
      for i in range(n):
        row = copy.deepcopy(low_stat_minimisation.x)
        row[i] += 0.1* initial_step_size * max(row[i], 1.0) # 10% of initial simplex
        initial_simplex.append(row)

      minimisation = minimize(func, low_stat_minimisation.x, method='Nelder-Mead', tol=0.1, options={'xatol': 0.001, 'fatol': 0.1, 'initial_simplex': initial_simplex})
      return minimisation.x, minimisation.fun