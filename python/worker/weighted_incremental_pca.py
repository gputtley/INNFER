import numpy as np
import joblib


class WeightedIncrementalPCA:
  """
  Weighted incremental PCA with a sklearn-like partial_fit API.

  Usage:
      pca = WeightedIncrementalPCA(n_components=n_features, whiten=True)

      for X_batch, w_batch in batches:
          pca.partial_fit(X_batch, sample_weight=w_batch)

      Z = pca.transform(X)
      X_back = pca.inverse_transform(Z)

  Notes:
      - For invertibility, use n_components == n_features.
      - sample_weight must be non-negative.
      - This accumulates the full weighted covariance matrix, so memory scales as
        O(n_features^2), not O(n_events).
  """

  def __init__(self, n_components=None, whiten=False, batch_size=None, eps=1e-12):
    self.n_components = n_components
    self.whiten = whiten
    self.batch_size = batch_size
    self.eps = eps

    self._is_fitted = False
    self._is_finalized = False

  def partial_fit(self, X, sample_weight=None):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
      raise ValueError("X must have shape (n_samples, n_features).")

    n_samples, n_features = X.shape

    if sample_weight is None:
      w = np.ones(n_samples, dtype=float)
    else:
      w = np.asarray(sample_weight, dtype=float)

    if w.ndim != 1:
      raise ValueError("sample_weight must be 1D.")

    if len(w) != n_samples:
      raise ValueError("sample_weight must have length n_samples.")

    if np.any(w < 0):
      raise ValueError("sample_weight must be non-negative.")

    batch_sum_w = np.sum(w)

    if batch_sum_w <= 0:
      return self

    batch_mean = np.sum(X * w[:, None], axis=0) / batch_sum_w
    Xc = X - batch_mean

    # Weighted within-batch scatter matrix
    batch_M2 = (Xc * w[:, None]).T @ Xc

    if not self._is_fitted:
      self.n_features_in_ = n_features
      self.sum_w_ = batch_sum_w
      self.mean_ = batch_mean
      self._M2 = batch_M2
      self._is_fitted = True
      self._is_finalized = False
      return self

    if n_features != self.n_features_in_:
      raise ValueError(
        f"X has {n_features} features, expected {self.n_features_in_}."
      )

    # Merge old accumulated stats with new batch stats.
    # This is the weighted parallel covariance update.
    old_sum_w = self.sum_w_
    new_sum_w = old_sum_w + batch_sum_w

    delta = batch_mean - self.mean_

    self._M2 = (
      self._M2
      + batch_M2
      + np.outer(delta, delta) * old_sum_w * batch_sum_w / new_sum_w
    )

    self.mean_ = self.mean_ + delta * batch_sum_w / new_sum_w
    self.sum_w_ = new_sum_w

    self._is_finalized = False

    return self

  def finalize(self):
    """
    Compute PCA components from accumulated weighted covariance.

    You can call this manually after all partial_fit calls, but transform()
    and inverse_transform() will call it automatically if needed.
    """
    if not self._is_fitted:
      raise RuntimeError("This PCA has not been fitted yet.")

    cov = self._M2 / self.sum_w_
    self.covariance_ = cov

    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort largest eigenvalue first, like sklearn PCA
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if self.n_components is None:
      n_components = self.n_features_in_
    else:
      n_components = self.n_components

    if n_components > self.n_features_in_:
      raise ValueError("n_components cannot be larger than n_features.")

    self.n_components_ = n_components

    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]

    self.explained_variance_ = np.maximum(eigvals, self.eps)

    # sklearn convention: components_.shape = (n_components, n_features)
    self.components_ = eigvecs.T

    total_variance = np.sum(np.maximum(np.linalg.eigvalsh(cov), self.eps))
    self.explained_variance_ratio_ = self.explained_variance_ / total_variance

    self._is_finalized = True

    return self

  def transform(self, X):
    if not self._is_finalized:
      self.finalize()

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
      raise ValueError("X must have shape (n_samples, n_features).")

    if X.shape[1] != self.n_features_in_:
      raise ValueError(
        f"X has {X.shape[1]} features, expected {self.n_features_in_}."
      )

    Xc = X - self.mean_
    Z = Xc @ self.components_.T

    if self.whiten:
      Z = Z / np.sqrt(self.explained_variance_)

    return Z

  def inverse_transform(self, Z):
    if not self._is_finalized:
      self.finalize()

    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2:
      raise ValueError("Z must have shape (n_samples, n_components).")

    if Z.shape[1] != self.n_components_:
      raise ValueError(
        f"Z has {Z.shape[1]} components, expected {self.n_components_}."
      )

    Y = Z.copy()

    if self.whiten:
      Y = Y * np.sqrt(self.explained_variance_)

    X_back = Y @ self.components_ + self.mean_

    return X_back

  def fit(self, X, sample_weight=None):
    self.partial_fit(X, sample_weight=sample_weight)
    self.finalize()
    return self

  def fit_transform(self, X, sample_weight=None):
    return self.fit(X, sample_weight=sample_weight).transform(X)

  def log_abs_det_jacobian(self):
    """
    log |det(dz/dx)| for density transformations.

    Only valid for full-rank PCA.
    """
    if not self._is_finalized:
      self.finalize()

    if self.n_components_ != self.n_features_in_:
      raise ValueError(
        "Jacobian determinant only valid for full-rank PCA: "
        "n_components == n_features."
      )

    if not self.whiten:
      return 0.0

    return -0.5 * np.sum(np.log(self.explained_variance_))

  def save(self, path):
    if not self._is_finalized:
      self.finalize()

    joblib.dump(self, path)

  @staticmethod
  def load(path):
    return joblib.load(path)