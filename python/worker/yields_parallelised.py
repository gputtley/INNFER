import numpy as np
import pandas as pd

class Yields:

  def __init__(
      self,
      nominal_yield,
      lnN = {},
      physics_model = None,
      rate_param = None, 
    ):

    if not isinstance(nominal_yield, list):
      nominal_yield = [nominal_yield]
    if not isinstance(lnN, list):
      lnN = [lnN]

    self.nominal_yields = np.asarray(nominal_yield, dtype=float)
    self.n_models = self.nominal_yields.size

    self.lnN_list = lnN or [{} for _ in range(self.n_models)]

    self.physics_model = physics_model
    self.rate_param = rate_param

    self.q = 0.5

    # Collect all nuisances
    nuisances = set()
    for d in self.lnN_list:
        nuisances.update(d.keys())

    self.nuisances = sorted(nuisances)

    # Fast lookup dictionary
    self.nuis_idx = {n: i for i, n in enumerate(self.nuisances)}
    n_nuis = len(self.nuisances)

    # Build kappa matrices
    self.kappa_lo = np.ones((self.n_models, n_nuis))
    self.kappa_hi = np.ones((self.n_models, n_nuis))

    for m, d in enumerate(self.lnN_list):
      for n, (lo, hi) in d.items():
        j = self.nuis_idx[n]
        self.kappa_lo[m, j] = lo
        self.kappa_hi[m, j] = hi

    # Precompute logs
    self.log_lo = np.log(self.kappa_lo)
    self.log_hi = np.log(self.kappa_hi)

    # constant used in interpolation
    self.avg = 0.5 * (self.log_hi - self.log_lo)
    self.sum_logs = self.log_hi + self.log_lo


  def GetYield(self, Y):

    if len(Y) == 0:
      return pd.DataFrame(
          [self.nominal_yields],
          columns=[f"yield_{i}" for i in range(self.n_models)]
        )

    Y = Y.reset_index(drop=True)

    n_rows = len(Y)

    # Start from nominal
    out = np.broadcast_to(self.nominal_yields, (n_rows, self.n_models)).copy()

    # lnN systematics
    if self.nuisances:

      cols = [c for c in self.nuisances if c in Y.columns]

      if cols:

        idx = np.array([self.nuis_idx[c] for c in cols])

        nu = Y[cols].values   # (rows, nuis)

        # Select only relevant nuisances
        log_lo = self.log_lo[:, idx]
        log_hi = self.log_hi[:, idx]
        avg = self.avg[:, idx]
        sum_logs = self.sum_logs[:, idx]

        nu3 = nu[:, None, :]  # (rows,1,nuis)

        high = nu3 >= self.q
        low = nu3 < -self.q
        mid = ~(high | low)

        alpha = 0.125 * (2 * nu3) * (
            (6 * nu3**4) - (10 * nu3**2) + 5
        )

        logk = (
            high * log_hi[None, :, :]
            + low * (-log_lo[None, :, :])
            + mid * (avg[None, :, :] + alpha * sum_logs[None, :, :])
        )

        out *= np.exp(np.sum(logk * nu3, axis=2))

    # rate parameter
    if self.rate_param and self.rate_param in Y.columns:
      out *= Y[self.rate_param].values[:, None]

    # physics model
    if self.physics_model is not None:
      out *= self.physics_model(Y)[:, None]

    if len(out) == 1:
      if len(out[0]) == 1:
        return out[0, 0]
      else:
        return list(out[0])
    else:
      if self.n_models == 1:
        return pd.DataFrame(out[:, 0], columns=["yield"])
      else:
        return [pd.DataFrame(out[:, i], columns=[f"yield"]) for i in range(self.n_models)]

  def PrintSummary(self):

    print("------------------------------------------------")
    print("Yields Summary")
    print("------------------------------------------------")

    for i in range(self.n_models):
      print(f"[{i}] nominal = {self.nominal_yields[i]}")
      print(f"     lnN = {self.lnN_list[i]}")

    print(f"physics_model = {self.physics_model}")
    print(f"rate_param = {self.rate_param}")

    print("------------------------------------------------")
