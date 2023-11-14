import numpy as np
from preprocess import PreProcess

class Likelihood():
  
  def __init__(self, models, type="unbinned", parameters={}, data_parameters={}):

    self.type = type # unbinned, unbinned_extended
    self.models = models
    self.parameters = parameters
    self.data_parameters = data_parameters
    self.Y_columns = self._MakeY()

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

      lkld_per_x += rate_param * pdf.Probability(X, np.array([list(Y)]), y_columns=self.Y_columns)

    if wts is not None:
      lkld_per_x *= wts.flatten()
    
    ln_lkld = np.log(lkld_per_x).sum()

    if return_ln:
      return ln_lkld
    else:
      return np.exp(ln_lkld)

  def UnbinnedExtended(self, X, Y, wts=None):
    print("- Building unbinned extended likelihood")