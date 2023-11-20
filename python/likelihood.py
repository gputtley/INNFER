import copy
import numpy as np
from scipy.optimize import minimize

class Likelihood():
  
  def __init__(self, models, type="unbinned", parameters={}, data_parameters={}):

    self.type = type # unbinned, unbinned_extended
    self.models = models
    self.parameters = parameters
    self.data_parameters = data_parameters
    self.Y_columns = self._MakeY()
    
    # saved parameters
    self.best_fit = None
    self.best_fit_nll = None

  def _MakeY(self):

    Y_columns = []
    if "rate_parameters" in self.parameters.keys():
      Y_columns += ["mu_"+rate_parameter for rate_parameter in self.parameters["rate_parameters"]]
    for _, data_params in self.data_parameters.items():
      Y_columns += [name for name in data_params["Y_columns"] if name not in Y_columns]

    return Y_columns


  def Run(self, X, Y, wts=None, return_ln=False):
    if self.type == "unbinned":
      lkld_val = self.Unbinned(X, Y, wts=wts, return_ln=return_ln)
    elif self.type == "unbinned_extended":
      lkld_val = self.UnbinnedExtended(X, Y, wts=wts, return_ln=return_ln)

    return lkld_val

  def Unbinned(self, X, Y, wts=None, return_ln=False):

    lkld_per_x = np.zeros(len(X))

    for name, pdf in self.models["pdfs"].items():

      if "mu_"+name in self.Y_columns:
        rate_param = Y[self.Y_columns.index("mu_"+name)]
      else:
        rate_param = 1.0

      # add constraints here

      lkld_per_x += rate_param * pdf.Probability(copy.deepcopy(X), np.array([list(Y)]), y_columns=self.Y_columns)

    if wts is not None:
      lkld_per_x = lkld_per_x**wts.flatten()
    
    ln_lkld = np.log(lkld_per_x).sum()

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

  def UnbinnedExtended(self, X, Y, wts=None):
    print("- Building unbinned extended likelihood")

  def GetBestFit(self, X, initial_guess, wts=None):
    def NLL(Y): 
      nll = -2*self.Run(X, Y, wts=wts, return_ln=True)
      #print(Y, nll)
      return nll
    result = self.Minimise(NLL, initial_guess)
    self.best_fit = result[0]
    self.best_fit_nll = result[1]

  def MakeScanInSeries(self, X, column, wts=None, estimated_sigmas_shown=3, estimated_sigma_step=0.2, initial_step_fraction=0.001):

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

  def FindCrossings(self, x, y, best_fit, crossings=[1, 2]):
    values = {}
    for crossing in crossings:
      for sign in [-1, 1]:
        condition = np.array(x) * sign > best_fit * sign
        filtered_x = np.array(x)[condition]
        filtered_y = np.array(y)[condition]
        sorted_indices = np.argsort(filtered_y)
        filtered_x = filtered_x[sorted_indices]
        filtered_y = filtered_y[sorted_indices]
        if crossing**2 > min(filtered_y) and crossing**2 < max(filtered_y):
          values[sign * crossing] = np.interp(crossing**2, filtered_y, filtered_x)
    crossings[0] = x[y.index(min(y))]
    return values

  def Minimise(self, func, initial_guess):
    minimisation = minimize(func, initial_guess, method='Nelder-Mead', tol=0.1, options={'xatol': 0.001, 'fatol': 0.1, 'maxiter': 20})
    return minimisation.x, minimisation.fun