"""Convert flow log probabilities to physical densities, including spline Jacobians."""
import hashlib
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import norm

from data_processor import DataProcessor


def UnTransformDensityProbability(log_probs, X, parameters, transform_X=True, cache=None, max_cached_batches=1024):

  if not transform_X:
    return log_probs.copy()

  columns = list(log_probs.columns)
  if cache is not None:
    metadata = hashlib.blake2b(pickle.dumps(parameters), digest_size=16).digest()
    if cache.get("metadata") != metadata:
      cache.clear()
      cache.update({"metadata": metadata, "columns": {}, "splines": {}, "batches": OrderedDict()})
    result = log_probs.copy()
    for column in columns:
      if column not in cache["columns"]:
        dp = DataProcessor([[pd.DataFrame({column: [0., 1.]})]], "dataset", options={"parameters": parameters})
        cache["columns"][column] = dp.UnTransformData(pd.DataFrame({column: [0., 1.]}))[column].to_numpy()
      zero, one = cache["columns"][column]
      result[column] = result[column] + zero if column == "log_prob" else result[column] * one
  else:
    dp = DataProcessor([[log_probs]], "dataset", options={"parameters": parameters})
    result = dp.UnTransformData(log_probs.copy()).loc[:, columns]

  # Observable preprocessing is independent of the conditions: only log p,
  # not its condition derivatives, receives this additional Jacobian.
  forward = {column: path for column, path in parameters.get("spline_to_gaussian", {}).get("forward", {}).items()
    if column in parameters["X_columns"]}
  if "log_prob" not in columns or not forward:
    return result
  if X is None:
    raise ValueError("Physical observables are required for spline density Jacobians")

  values = np.ascontiguousarray(X.loc[:, list(forward)].to_numpy())
  key = (tuple(forward), values.shape, values.dtype.str, hashlib.blake2b(values.tobytes(), digest_size=16).digest())
  if cache is not None and key in cache["batches"]:
    cache["batches"].move_to_end(key)
    jacobian = cache["batches"][key]
  else:
    jacobian = np.zeros(len(X), dtype=np.float64)
    for column, path in forward.items():
      if cache is not None and path in cache["splines"]:
        spline = cache["splines"][path]
      else:
        with open(path, "rb") as file:
          spline = pickle.load(file)
        if cache is not None:
          cache["splines"][path] = spline
      x = X[column].to_numpy()
      u, slope = spline(x), spline.derivative()(x)
      valid = (u > 0.) & (u < 1.) & (slope > 0.) & np.isfinite(u) & np.isfinite(slope)
      term = np.full(len(X), -np.inf)
      term[valid] = np.log(slope[valid]) - norm.logpdf(norm.ppf(u[valid]))
      jacobian += term
    if cache is not None:
      cache["batches"][key] = jacobian
      while len(cache["batches"]) > max_cached_batches:
        cache["batches"].popitem(last=False)

  result["log_prob"] = np.where(np.isneginf(jacobian), -np.inf, result["log_prob"].to_numpy() + jacobian)
  return result
