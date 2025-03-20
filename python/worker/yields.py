import numpy as np
import pandas as pd

class Yields():

  def __init__(
      self, 
      nominal_yield,
      lnN = {},
      physics_model = None,
      rate_param = None, 
    ):
    self.nominal_yield = nominal_yield
    self.lnN = lnN
    self.physics_model = physics_model
    self.rate_param = rate_param
    self.q = 0.5

  def _LogAsymKappa(self, nu, kappa_low, kappa_high):

    log_asym_kappa = np.ones(len(nu))

    log_kappa_low = np.log(kappa_low)
    log_kappa_high = np.log(kappa_high)
    
    nu_high_indices = (nu >= self.q)
    nu_low_indices = (nu < -self.q)
    nu_middle_indices = ((nu >= -self.q) & (nu < self.q))
    
    log_asym_kappa[nu_high_indices] = log_kappa_high
    log_asym_kappa[nu_low_indices] = -log_kappa_low

    avg = 0.5 * (log_kappa_high - log_kappa_low)
    alpha = 0.125 * (2 * nu[nu_middle_indices]) * ((6 * nu[nu_middle_indices]**4) - (10 * nu[nu_middle_indices]**2) + 5)
    log_asym_kappa[nu_middle_indices] = avg + alpha * (log_kappa_high + log_kappa_low)

    return log_asym_kappa


  def GetYield(self, Y, ignore_rate_param=False, ignore_physics_model=False, ignore_lnN=False):

    # Reindex Y
    Y = Y.reset_index(drop=True)

    if len(Y) == 0:
      # Fix empty Y
      out = pd.DataFrame({"yield" : [self.nominal_yield]})
    else:
      # Set yield to nominal yield using a dataframe the same size as Y
      out = pd.DataFrame(self.nominal_yield*np.ones(len(Y)), columns=["yield"])

    # do rate parameter
    if not ignore_rate_param:
      if self.rate_param is not None:
        if self.rate_param in Y.columns:
          out.loc[:,"yield"] *= Y.loc[:,self.rate_param]

    # do physics model
    if not ignore_physics_model:
      if self.physics_model is not None:
        out.loc[:,"yield"] *= self.physics_model(Y)

    # do lnN
    if not ignore_lnN:
      for k, v in self.lnN.items():
        if k in Y.columns:
          out.loc[:,"yield"] *= np.exp(self._LogAsymKappa(Y.loc[:,k].to_numpy(), v[0], v[1])) ** Y.loc[:,k].to_numpy()

    # return
    if len(out) == 1:
      return float(out.loc[0,"yield"])
    else:
      return out
    
  def PrintSummary(self):
    print("----------------------------------------------------------------")
    print("Yields Summary")
    print("----------------------------------------------------------------")
    print(f"nominal_yield = {self.nominal_yield}")
    print(f"lnN = {self.lnN}")
    print(f"physics_model = {self.physics_model}")
    print(f"rate_param = {self.rate_param}")
    print("----------------------------------------------------------------")
