import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

import numpy as np
import pandas as pd

from functools import partial
from pandas.errors import PerformanceWarning
from sklearn.metrics import auc as roc_curve_auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor
from useful_functions import Resample

warnings.simplefilter("ignore", PerformanceWarning)

def SignedClassifierMetrics(labels, scores, weights):
  """Signed two-sample score statistics; calibrate with signed-weight toys."""
  labels = np.asarray(labels)
  scores = np.asarray(scores, dtype=float)
  weights = np.asarray(weights, dtype=float)
  if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights)):
    raise ValueError("Non-finite classifier scores or weights")
  if not np.all(np.isin(labels, [0, 1])):
    raise ValueError("Signed classifier metrics require binary labels")
  sums = [weights[labels == k].sum() for k in [0, 1]]
  if min(sums) <= 0:
    raise ValueError("Each classifier population needs positive signed total weight")
  order = np.argsort(scores, kind="stable")
  sorted_scores, sorted_labels, sorted_weights = scores[order], labels[order], weights[order]
  _, starts = np.unique(sorted_scores, return_index=True)
  w0 = np.add.reduceat(np.where(sorted_labels == 0, sorted_weights, 0.), starts)
  w1 = np.add.reduceat(np.where(sorted_labels == 1, sorted_weights, 0.), starts)
  # Count weighted pairs, giving tied scores half credit.
  auc = np.sum(w1 * (np.cumsum(w0) - 0.5 * w0)) / (sums[0] * sums[1])
  correct = (scores >= 0).astype(int) == labels
  accuracy = 0.5 * sum(weights[(labels == k) & correct].sum() / sums[k] for k in [0, 1])
  return float(auc), float(accuracy)


class MultiDimMetrics():

  def __init__(self, sim_files, synth_files, columns, sim_fraction=1.0, synth_fraction=1.0, functions_to_apply=[], sim_wt_name="wt", synth_wt_name="wt", sim_selection=None, synth_selection=None):
    self.sim_files = sim_files
    self.synth_files = synth_files
    self.sim_fraction = sim_fraction
    self.synth_fraction = synth_fraction
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    self.verbose = True
    self.wasserstein_slices = 100
    self.resample_sim = False
    self.columns = columns
    self.functions_to_apply = functions_to_apply
    self.sim_wt_name = sim_wt_name
    self.synth_wt_name = synth_wt_name
    self.sim_selection = sim_selection
    self.synth_selection = synth_selection

    self.sim_dataset = None
    self.synth_dataset = None
    self.sim_train = None
    self.sim_test = None
    self.synth_train = None
    self.synth_test = None
    self.metrics = []
    self.signed_bdt_metrics = {}

  def _BinNegativeWeightedEvents(self, X, wt):

    # Sort by X
    indices = np.argsort(X)
    X = X[indices]
    wt = wt[indices]

    # find negative weights
    neg_wt_indices = wt < 0
    count_neg_wts = np.sum(neg_wt_indices)

    # while statement to bin negative weights
    break_after = 1000
    ind = 0
    while count_neg_wts > 0 and ind < break_after:

      # Increase index
      ind += 1

      # Find negative weights
      neg_indices = np.where(neg_wt_indices)[0]

      # Index to sum with - add 1 to neg_indices
      closest_X_index = neg_indices + 1

      # if final then add to one before
      if closest_X_index[-1] == len(X):
        closest_X_index[-1] = neg_indices[-1] - 1

      # Update values
      X[closest_X_index] = (X[closest_X_index]+ X[neg_indices])/2
      wt[closest_X_index] += wt[neg_indices]

      # Remove negative weights
      X = X[~neg_wt_indices]
      wt = wt[~neg_wt_indices]

      # Sort by X
      neg_wt_indices = wt < 0
      count_neg_wts = np.sum(neg_wt_indices)

    return X, wt


  def _GetWasserstein(self, data1, weights1, data2, weights2):

    # Integrate empirical step CDFs, rather than interpolating between samples.
    # Signed weights give a CDF discrepancy, not a probability Wasserstein metric.
    data1, data2 = np.asarray(data1), np.asarray(data2)
    weights1, weights2 = np.asarray(weights1), np.asarray(weights2)
    if weights1.sum() <= 0 or weights2.sum() <= 0:
      raise ValueError("CDF comparison requires positive signed total weights")
    support = np.unique(np.concatenate([data1, data2]))
    if len(support) < 2:
      return 0.0
    def cdf(data, weights):
      order = np.argsort(data)
      cumulative = np.r_[0., np.cumsum(weights[order]) / weights.sum()]
      return cumulative[np.searchsorted(data[order], support[:-1], side="right")]
    return float(np.sum(np.abs(cdf(data1, weights1) - cdf(data2, weights2)) * np.diff(support)))

  def AddBDTSeparation(self):
    self.metrics.append("BDT Separation")

  def AddKMeansChiSquared(self):
    self.metrics.append("K-Means Chi Squared")

  def AddWassersteinSliced(self):
    self.metrics.append("Wasserstein Sliced")

  def AddWassersteinUnbinned(self):
    self.metrics.append("Wasserstein Unbinned")

  """
  def DoBDTSeparation(self, remove_neg_weights=False, no_conditions=True, only_conditions=False):

    if self.verbose:
      print(f" - Doing BDT separation")

    import xgboost as xgb

    # Make training and testing datasets
    train_columns = self.columns
    if self.sim_train is None:
      sim = self.sim_dataset.copy()
      synth = self.synth_dataset.copy()
      sim.loc[:, "y"] = 0.0
      synth.loc[:, "y"] = 1.0
      total = pd.concat([synth, sim], ignore_index=True)
      total = total.sample(frac=1).reset_index(drop=True)
      del sim, synth
      X_wt_train, X_wt_test, y_train, y_test = train_test_split(total.loc[:, train_columns + ["wt"]], total.loc[:,"y"], test_size=0.5, random_state=42)
      wt_train = X_wt_train.loc[:,"wt"].to_numpy()
      wt_test = X_wt_test.loc[:,"wt"].to_numpy()
      X_train = X_wt_train.loc[:,train_columns].to_numpy()
      X_test = X_wt_test.loc[:,train_columns].to_numpy()
      y_train = y_train.to_numpy()
      y_test = y_test.to_numpy()
      del X_wt_train, X_wt_test, total
    else:
      self.sim_train["y"] = 0.0
      self.synth_train["y"] = 1.0
      self.sim_test["y"] = 0.0
      self.synth_test["y"] = 1.0

      total_train = pd.concat([self.sim_train, self.synth_train], ignore_index=True)
      total_test = pd.concat([self.sim_test, self.synth_test], ignore_index=True)
      total_train = total_train.sample(frac=1).reset_index(drop=True)
      total_test = total_test.sample(frac=1).reset_index(drop=True)
      wt_train = total_train.loc[:,"wt"].to_numpy()
      wt_test = total_test.loc[:,"wt"].to_numpy()
      X_train = total_train.loc[:,train_columns].to_numpy()
      X_test = total_test.loc[:,train_columns].to_numpy()
      y_train = total_train.loc[:,"y"].to_numpy()
      y_test = total_test.loc[:,"y"].to_numpy()
      del total_train, total_test
      

    # Resample sim dataset
    if self.resample_sim:
      
      # Get indices
      indices_train_0 = (y_train==0)
      indices_train_1 = (y_train==1)
      indices_test_0 = (y_test==0)
      indices_test_1 = (y_test==1)
      # Get sim
      X_sim_train = X_train[indices_train_0]
      y_sim_train = y_train[indices_train_0]
      wt_sim_train = wt_train[indices_train_0]
      X_sim_test = X_test[indices_test_0]
      y_sim_test = y_test[indices_test_0]
      wt_sim_test = wt_test[indices_test_0]
      # Combine X and y
      Xy_sim_train = np.concatenate([X_sim_train, y_sim_train.reshape(-1,1)], axis=1)
      Xy_sim_test = np.concatenate([X_sim_test, y_sim_test.reshape(-1,1)], axis=1)
      # Resample
      Xy_sim_train, wt_sim_train = Resample(Xy_sim_train, wt_sim_train, method="oversample", keep_weights=False, sample_size="length", total_scale="sum_weights")
      Xy_sim_test, wt_sim_test = Resample(Xy_sim_test, wt_sim_test, method="oversample", keep_weights=False, sample_size="length", total_scale="sum_weights")
      # Split back
      X_sim_train = Xy_sim_train[:,:-1]
      y_sim_train = Xy_sim_train[:,-1]
      X_sim_test = Xy_sim_test[:,:-1]
      y_sim_test = Xy_sim_test[:,-1]
      del Xy_sim_train, Xy_sim_test
      # Make full train and test
      X_train = np.concatenate([X_sim_train, X_train[indices_train_1]], axis=0)
      y_train = np.concatenate([y_sim_train, y_train[indices_train_1]], axis=0)
      wt_train = np.concatenate([wt_sim_train, wt_train[indices_train_1]], axis=0)
      X_test = np.concatenate([X_sim_test, X_test[indices_test_1]], axis=0)
      y_test = np.concatenate([y_sim_test, y_test[indices_test_1]], axis=0)
      wt_test = np.concatenate([wt_sim_test, wt_test[indices_test_1]], axis=0)
      del X_sim_train, X_sim_test, y_sim_train, y_sim_test, wt_sim_train, wt_sim_test
      # Shuffle
      indices = np.random.permutation(len(X_train))
      X_train = X_train[indices]
      y_train = y_train[indices]
      wt_train = wt_train[indices]
      indices = np.random.permutation(len(X_test))
      X_test = X_test[indices]
      y_test = y_test[indices]
      wt_test = wt_test[indices]
      del indices

    # normalise weights and scale to eff events in train and 1 in test
    sum_wt_0 = np.sum(wt_train[y_train==0])
    sum_wt_1 = np.sum(wt_train[y_train==1])
    eff_events_0 = sum_wt_0**2 / np.sum(wt_train[y_train==0]**2)
    wt_train[y_train==0] *= eff_events_0/sum_wt_0
    wt_train[y_train==1] *= eff_events_0/sum_wt_1
    wt_test[y_test==0] /= sum_wt_0
    wt_test[y_test==1] /= sum_wt_1

    # print the fraction of negitive weights in each class
    frac_neg_train_0 = np.sum((wt_train<0) & (y_train==0)) / np.sum(y_train==0)
    frac_neg_train_1 = np.sum((wt_train<0) & (y_train==1)) / np.sum(y_train==1)

    # No negative weights
    if remove_neg_weights:
      train_indices = (wt_train>0)
      X_train = X_train[train_indices]
      y_train = y_train[train_indices]
      wt_train = wt_train[train_indices]
      test_indices = (wt_test>0)
      X_test = X_test[test_indices]
      y_test = y_test[test_indices]
      wt_test = wt_test[test_indices]
    # Do fix if negative weights
    else:
      cat_num = 2
      # train
      # y == 0
      neg_train_wt_inds_0 = ((wt_train<0) & (y_train==0))
      neg_train_wt_0 = len(wt_train[neg_train_wt_inds_0]) > 0
      neg_wt_0 = None
      if neg_train_wt_0:
        neg_wt_0 = 1*cat_num
        y_train[neg_train_wt_inds_0] = neg_wt_0
        wt_train[neg_train_wt_inds_0] *= -1
        cat_num += 1
      # y == 1
      neg_train_wt_inds_1 = ((wt_train<0) & (y_train==1))
      neg_train_wt_1 = len(wt_train[neg_train_wt_inds_1]) > 0
      neg_wt_1 = None
      if neg_train_wt_1:
        neg_wt_1 = 1*cat_num
        y_train[neg_train_wt_inds_1] = neg_wt_1
        wt_train[neg_train_wt_inds_1] *= -1
      # test
      # y == 0
      neg_test_wt_inds_0 = ((wt_test<0) & (y_test==0))
      neg_test_wt_0 = len(wt_test[neg_test_wt_inds_0]) > 0
      neg_wt_0 = None
      if neg_test_wt_0:
        neg_wt_0 = 1*cat_num
        y_test[neg_test_wt_inds_0] = neg_wt_0
        wt_test[neg_test_wt_inds_0] *= -1
      # y == 1
      neg_test_wt_inds_1 = ((wt_test<0) & (y_test==1))
      neg_test_wt_1 = len(wt_test[neg_test_wt_inds_1]) > 0
      neg_wt_1 = None
      if neg_test_wt_1:
        neg_wt_1 = 1*cat_num
        y_test[neg_test_wt_inds_1] = neg_wt_1
        wt_test[neg_test_wt_inds_1] *= -1

    # Train separator
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train, sample_weight=wt_train)

    # Get probabilities
    train_proba = clf.predict_proba(X_train)
    test_proba = clf.predict_proba(X_test)

    if not remove_neg_weights:
      if neg_wt_0 is None:
        y_prob_sim_train = train_proba[:,0]
        y_prob_sim_test = test_proba[:,0]
      else:
        y_prob_sim_train = train_proba[:,0] - train_proba[:,neg_wt_0]
        y_prob_sim_test = test_proba[:,0] - test_proba[:,neg_wt_0]
        y_train[y_train == neg_wt_0] = 0
        y_test[y_test == neg_wt_0] = 0
      if neg_wt_1 is None:
        y_prob_synth_train = train_proba[:,1]
        y_prob_synth_test = test_proba[:,1]
      else:
        y_prob_synth_train = train_proba[:,1] - train_proba[:,neg_wt_1]
        y_prob_synth_test = test_proba[:,1] - test_proba[:,neg_wt_1]
        y_train[y_train == neg_wt_1] = 1
        y_test[y_test == neg_wt_1] = 1
      y_prob_train = y_prob_synth_train / (y_prob_synth_train + y_prob_sim_train)
      y_prob_test = y_prob_synth_test / (y_prob_synth_test + y_prob_sim_test)
    else:
      y_prob_train = train_proba[:,1]
      y_prob_test = test_proba[:,1]

    # normalise weights back down to 1 in each category
    wt_train[y_train==0] /= eff_events_0
    wt_train[y_train==1] /= eff_events_0

    # Get train auc
    fpr, tpr, thresholds = roc_curve(y_train, y_prob_train, sample_weight=wt_train)
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    train_auc = roc_curve_auc(fpr, tpr)

    # Get train accuracy
    y_pred_train = (y_prob_train >= 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, y_pred_train, sample_weight=wt_train)

    # Get test auc
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test, sample_weight=wt_test)
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    test_auc = roc_curve_auc(fpr, tpr)

    # Get test accuracy
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_test, sample_weight=wt_test)

    if self.verbose:
      print(f" - Train AUC: {train_auc}")
      print(f" - Test AUC: {test_auc}")
      print(f" - Train Accuracy: {train_accuracy}")
      print(f" - Test Accuracy: {test_accuracy}")

    return float(test_auc), float(test_accuracy)
  """

  def DoBDTSeparation(
      self,
      remove_neg_weights=False,
  ):
      """
      Train a BDT to distinguish simulated and synthetic events.

      Negative training weights are represented using auxiliary classifier
      classes:

        0: positive-weight simulated events
        1: positive-weight synthetic events
        2: negative-weight simulated events, if present
        3: negative-weight synthetic events, if present

      The auxiliary classes are used only during classifier training. Evaluation
      labels always remain binary.

      When negative weights are retained, sklearn-compatible AUC and accuracy are
      evaluated with absolute weights. Consequently, these metrics describe
      separation under the absolute-weight event measure, not a signed ROC.

      Parameters
      ----------
      remove_neg_weights : bool
          If True, remove negative-weight events from both training and testing.
          If False, use auxiliary training classes for negative-weight events.

      Returns
      -------
      tuple[float, float]
          Test AUC and test accuracy.
      """

      import xgboost as xgb

      if self.verbose:
          print(" - Doing BDT separation")

      def _negative_fraction(weights, labels, class_label):
          class_mask = labels == class_label
          class_count = np.count_nonzero(class_mask)

          if class_count == 0:
              return 0.0

          return float(
              np.count_nonzero((weights < 0) & class_mask) / class_count
          )

      def _check_binary_classes(labels, dataset_name):
          classes = np.unique(labels)

          if not np.array_equal(classes, np.array([0.0, 1.0])):
              raise ValueError(
                  f"{dataset_name} must contain binary labels 0 and 1; "
                  f"found {classes.tolist()}"
              )

      def _balance_evaluation_weights(weights, labels, dataset_name):
          """
          Convert to positive evaluation weights and give each binary class
          unit total weight.
          """
          balanced_weights = np.abs(np.asarray(weights, dtype=float)).copy()

          if not np.all(np.isfinite(balanced_weights)):
              raise ValueError(
                  f"{dataset_name} contains non-finite evaluation weights"
              )

          for class_label in (0.0, 1.0):
              class_mask = labels == class_label
              class_sum = np.sum(balanced_weights[class_mask])

              if not np.isfinite(class_sum) or class_sum <= 0:
                  raise ValueError(
                      f"{dataset_name} class {class_label:g} has invalid "
                      f"absolute weight sum {class_sum}"
                  )

              balanced_weights[class_mask] /= class_sum

          return balanced_weights

      def _probability_for_class(probabilities, class_to_column, class_label):
          if class_label is None:
              return np.zeros(probabilities.shape[0], dtype=float)

          if class_label not in class_to_column:
              raise ValueError(
                  f"Classifier output does not contain class {class_label}. "
                  f"Available classes are {sorted(class_to_column)}"
              )

          return probabilities[:, class_to_column[class_label]]

      train_columns = list(self.columns)

      # ------------------------------------------------------------------
      # Construct training and testing datasets
      # ------------------------------------------------------------------
      if self.sim_train is None:
          sim = self.sim_dataset.copy()
          synth = self.synth_dataset.copy()

          sim.loc[:, "y"] = 0.0
          synth.loc[:, "y"] = 1.0

          total = pd.concat([synth, sim], ignore_index=True)
          total = total.sample(frac=1).reset_index(drop=True)

          X_wt_train, X_wt_test, y_train, y_test = train_test_split(
              total.loc[:, train_columns + ["wt"]],
              total.loc[:, "y"],
              test_size=0.5,
              random_state=42,
              stratify=total.loc[:, "y"],
          )

          wt_train = X_wt_train.loc[:, "wt"].to_numpy(dtype=float)
          wt_test = X_wt_test.loc[:, "wt"].to_numpy(dtype=float)

          X_train = X_wt_train.loc[:, train_columns].to_numpy()
          X_test = X_wt_test.loc[:, train_columns].to_numpy()

          y_train = y_train.to_numpy(dtype=float)
          y_test = y_test.to_numpy(dtype=float)

          del sim
          del synth
          del total
          del X_wt_train
          del X_wt_test

      else:
          # Work on copies so this method does not mutate cached datasets.
          sim_train = self.sim_train.copy()
          synth_train = self.synth_train.copy()
          sim_test = self.sim_test.copy()
          synth_test = self.synth_test.copy()

          sim_train.loc[:, "y"] = 0.0
          synth_train.loc[:, "y"] = 1.0
          sim_test.loc[:, "y"] = 0.0
          synth_test.loc[:, "y"] = 1.0

          total_train = pd.concat(
              [sim_train, synth_train],
              ignore_index=True,
          )
          total_test = pd.concat(
              [sim_test, synth_test],
              ignore_index=True,
          )

          total_train = total_train.sample(frac=1).reset_index(drop=True)
          total_test = total_test.sample(frac=1).reset_index(drop=True)

          wt_train = total_train.loc[:, "wt"].to_numpy(dtype=float)
          wt_test = total_test.loc[:, "wt"].to_numpy(dtype=float)

          X_train = total_train.loc[:, train_columns].to_numpy()
          X_test = total_test.loc[:, train_columns].to_numpy()

          y_train = total_train.loc[:, "y"].to_numpy(dtype=float)
          y_test = total_test.loc[:, "y"].to_numpy(dtype=float)

          del sim_train
          del synth_train
          del sim_test
          del synth_test
          del total_train
          del total_test

      _check_binary_classes(y_train, "Training dataset")
      _check_binary_classes(y_test, "Testing dataset")

      if not np.all(np.isfinite(wt_train)):
          raise ValueError("Training weights contain non-finite values")

      if not np.all(np.isfinite(wt_test)):
          raise ValueError("Testing weights contain non-finite values")

      # ------------------------------------------------------------------
      # Optionally resample the simulated component
      # ------------------------------------------------------------------
      if self.resample_sim:
          indices_train_0 = y_train == 0.0
          indices_train_1 = y_train == 1.0
          indices_test_0 = y_test == 0.0
          indices_test_1 = y_test == 1.0

          X_sim_train = X_train[indices_train_0]
          y_sim_train = y_train[indices_train_0]
          wt_sim_train = wt_train[indices_train_0]

          X_sim_test = X_test[indices_test_0]
          y_sim_test = y_test[indices_test_0]
          wt_sim_test = wt_test[indices_test_0]

          Xy_sim_train = np.concatenate(
              [X_sim_train, y_sim_train.reshape(-1, 1)],
              axis=1,
          )
          Xy_sim_test = np.concatenate(
              [X_sim_test, y_sim_test.reshape(-1, 1)],
              axis=1,
          )

          Xy_sim_train, wt_sim_train = Resample(
              Xy_sim_train,
              wt_sim_train,
              method="oversample",
              keep_weights=False,
              sample_size="length",
              total_scale="sum_weights",
          )
          Xy_sim_test, wt_sim_test = Resample(
              Xy_sim_test,
              wt_sim_test,
              method="oversample",
              keep_weights=False,
              sample_size="length",
              total_scale="sum_weights",
          )

          X_sim_train = Xy_sim_train[:, :-1]
          y_sim_train = Xy_sim_train[:, -1]

          X_sim_test = Xy_sim_test[:, :-1]
          y_sim_test = Xy_sim_test[:, -1]

          X_train = np.concatenate(
              [X_sim_train, X_train[indices_train_1]],
              axis=0,
          )
          y_train = np.concatenate(
              [y_sim_train, y_train[indices_train_1]],
              axis=0,
          )
          wt_train = np.concatenate(
              [wt_sim_train, wt_train[indices_train_1]],
              axis=0,
          )

          X_test = np.concatenate(
              [X_sim_test, X_test[indices_test_1]],
              axis=0,
          )
          y_test = np.concatenate(
              [y_sim_test, y_test[indices_test_1]],
              axis=0,
          )
          wt_test = np.concatenate(
              [wt_sim_test, wt_test[indices_test_1]],
              axis=0,
          )

          del Xy_sim_train
          del Xy_sim_test
          del X_sim_train
          del X_sim_test
          del y_sim_train
          del y_sim_test
          del wt_sim_train
          del wt_sim_test

          train_permutation = np.random.permutation(len(X_train))
          X_train = X_train[train_permutation]
          y_train = y_train[train_permutation]
          wt_train = wt_train[train_permutation]

          test_permutation = np.random.permutation(len(X_test))
          X_test = X_test[test_permutation]
          y_test = y_test[test_permutation]
          wt_test = wt_test[test_permutation]

      _check_binary_classes(y_train, "Resampled training dataset")
      _check_binary_classes(y_test, "Resampled testing dataset")

      # Preserve binary labels. y_train may subsequently be changed to contain
      # auxiliary negative-weight classes; y_test must remain binary.
      y_train_binary = y_train.copy()
      y_test_binary = y_test.copy()

      # ------------------------------------------------------------------
      # Report negative-weight fractions before any filtering or sign changes
      # ------------------------------------------------------------------
      frac_neg_train_0 = _negative_fraction(
          wt_train,
          y_train_binary,
          0.0,
      )
      frac_neg_train_1 = _negative_fraction(
          wt_train,
          y_train_binary,
          1.0,
      )
      frac_neg_test_0 = _negative_fraction(
          wt_test,
          y_test_binary,
          0.0,
      )
      frac_neg_test_1 = _negative_fraction(
          wt_test,
          y_test_binary,
          1.0,
      )

      if self.verbose:
          print(
              " - Negative-weight fractions:"
              f" train sim={frac_neg_train_0:.6f},"
              f" train synth={frac_neg_train_1:.6f},"
              f" test sim={frac_neg_test_0:.6f},"
              f" test synth={frac_neg_test_1:.6f}"
          )

      # ------------------------------------------------------------------
      # Balance the two training populations
      # ------------------------------------------------------------------
      train_sim_mask = y_train_binary == 0.0
      train_synth_mask = y_train_binary == 1.0

      sum_wt_0 = np.sum(wt_train[train_sim_mask])
      sum_wt_1 = np.sum(wt_train[train_synth_mask])
      sum_wt_squared_0 = np.sum(wt_train[train_sim_mask] ** 2)

      if not np.isfinite(sum_wt_0) or sum_wt_0 <= 0:
          raise ValueError(
              f"Simulated training sample has invalid signed weight sum "
              f"{sum_wt_0}"
          )

      if not np.isfinite(sum_wt_1) or sum_wt_1 <= 0:
          raise ValueError(
              f"Synthetic training sample has invalid signed weight sum "
              f"{sum_wt_1}"
          )

      if not np.isfinite(sum_wt_squared_0) or sum_wt_squared_0 <= 0:
          raise ValueError(
              "Simulated training sample has invalid sum of squared weights"
          )

      eff_events_0 = sum_wt_0**2 / sum_wt_squared_0

      if not np.isfinite(eff_events_0) or eff_events_0 <= 0:
          raise ValueError(
              f"Invalid effective simulated event count {eff_events_0}"
          )

      # Preserve the original behaviour: give both training populations the
      # same signed total weight, equal to the simulated effective event count.
      wt_train[train_sim_mask] *= eff_events_0 / sum_wt_0
      wt_train[train_synth_mask] *= eff_events_0 / sum_wt_1

      # ------------------------------------------------------------------
      # Handle negative weights
      # ------------------------------------------------------------------
      neg_class_sim = None
      neg_class_synth = None

      if remove_neg_weights:
          train_keep = wt_train > 0
          test_keep = wt_test > 0

          X_train = X_train[train_keep]
          y_train = y_train[train_keep]
          y_train_binary = y_train_binary[train_keep]
          wt_train = wt_train[train_keep]

          X_test = X_test[test_keep]
          y_test_binary = y_test_binary[test_keep]
          wt_test = wt_test[test_keep]

          _check_binary_classes(
              y_train_binary,
              "Positive-weight training dataset",
          )
          _check_binary_classes(
              y_test_binary,
              "Positive-weight testing dataset",
          )

      else:
          next_class = 2

          neg_train_sim = (
              (wt_train < 0)
              & (y_train_binary == 0.0)
          )
          if np.any(neg_train_sim):
              neg_class_sim = next_class
              next_class += 1

              y_train[neg_train_sim] = neg_class_sim
              wt_train[neg_train_sim] *= -1

          neg_train_synth = (
              (wt_train < 0)
              & (y_train_binary == 1.0)
          )
          if np.any(neg_train_synth):
              neg_class_synth = next_class
              next_class += 1

              y_train[neg_train_synth] = neg_class_synth
              wt_train[neg_train_synth] *= -1

          # A negative component present only in the test split cannot be
          # represented by the classifier trained above.
          neg_test_sim = (
              (wt_test < 0)
              & (y_test_binary == 0.0)
          )
          neg_test_synth = (
              (wt_test < 0)
              & (y_test_binary == 1.0)
          )

          if np.any(neg_test_sim) and neg_class_sim is None:
              raise ValueError(
                  "Negative simulated events occur in the test split but not "
                  "the training split. The auxiliary negative class cannot be "
                  "trained. Use a larger/stratified split or remove negative "
                  "weights."
              )

          if np.any(neg_test_synth) and neg_class_synth is None:
              raise ValueError(
                  "Negative synthetic events occur in the test split but not "
                  "the training split. The auxiliary negative class cannot be "
                  "trained. Use a larger/stratified split or remove negative "
                  "weights."
              )

      if np.any(wt_train < 0):
          raise RuntimeError(
              "Negative XGBoost training weights remain after preprocessing"
          )

      if len(np.unique(y_train)) < 2:
          raise ValueError(
              "Fewer than two classifier training classes remain"
          )

      # ------------------------------------------------------------------
      # Train classifier
      # ------------------------------------------------------------------
      clf = xgb.XGBClassifier()
      clf.fit(
          X_train,
          y_train.astype(int),
          sample_weight=wt_train,
      )

      train_proba = clf.predict_proba(X_train)
      test_proba = clf.predict_proba(X_test)

      class_to_column = {
          int(class_label): column_index
          for column_index, class_label in enumerate(clf.classes_)
      }

      # ------------------------------------------------------------------
      # Construct the signed classifier score
      # ------------------------------------------------------------------
      p_sim_train = (
          _probability_for_class(
              train_proba,
              class_to_column,
              0,
          )
          - _probability_for_class(
              train_proba,
              class_to_column,
              neg_class_sim,
          )
      )
      p_synth_train = (
          _probability_for_class(
              train_proba,
              class_to_column,
              1,
          )
          - _probability_for_class(
              train_proba,
              class_to_column,
              neg_class_synth,
          )
      )

      p_sim_test = (
          _probability_for_class(
              test_proba,
              class_to_column,
              0,
          )
          - _probability_for_class(
              test_proba,
              class_to_column,
              neg_class_sim,
          )
      )
      p_synth_test = (
          _probability_for_class(
              test_proba,
              class_to_column,
              1,
          )
          - _probability_for_class(
              test_proba,
              class_to_column,
              neg_class_synth,
          )
      )

      # A difference is finite even when signed density estimates make
      # p_sim + p_synth zero or negative.
      score_train = p_synth_train - p_sim_train
      score_test = p_synth_test - p_sim_test

      if not np.all(np.isfinite(score_train)):
          raise ValueError("Training classifier scores contain NaN or infinity")

      if not np.all(np.isfinite(score_test)):
          raise ValueError("Testing classifier scores contain NaN or infinity")

      # ------------------------------------------------------------------
      # Prepare positive, class-balanced evaluation weights
      # ------------------------------------------------------------------
      wt_train_eval = _balance_evaluation_weights(
          wt_train,
          y_train_binary,
          "Training dataset",
      )
      wt_test_eval = _balance_evaluation_weights(
          wt_test,
          y_test_binary,
          "Testing dataset",
      )

      # ------------------------------------------------------------------
      # Calculate train metrics
      # ------------------------------------------------------------------
      fpr, tpr, _ = roc_curve(
          y_train_binary,
          score_train,
          sample_weight=wt_train_eval,
      )
      train_auc = roc_curve_auc(fpr, tpr)

      y_pred_train = (score_train >= 0.0).astype(int)
      train_accuracy = accuracy_score(
          y_train_binary,
          y_pred_train,
          sample_weight=wt_train_eval,
      )

      # ------------------------------------------------------------------
      # Calculate test metrics
      # ------------------------------------------------------------------
      fpr, tpr, _ = roc_curve(
          y_test_binary,
          score_test,
          sample_weight=wt_test_eval,
      )
      test_auc = roc_curve_auc(fpr, tpr)

      y_pred_test = (score_test >= 0.0).astype(int)
      test_accuracy = accuracy_score(
          y_test_binary,
          y_pred_test,
          sample_weight=wt_test_eval,
      )

      if self.verbose:
          print(f" - Train AUC: {train_auc}")
          print(f" - Test AUC: {test_auc}")
          print(f" - Train Accuracy: {train_accuracy}")
          print(f" - Test Accuracy: {test_accuracy}")

      signed_auc, signed_accuracy = SignedClassifierMetrics(
          y_test_binary, score_test, wt_test,
      )
      self.signed_bdt_metrics = {
          "bdt_signed_auc": signed_auc,
          "bdt_signed_accuracy": signed_accuracy,
      }

      return float(test_auc), float(test_accuracy)



  def DoKMeansChiSquared(self, n_clusters=50, n_init=10):

    if self.verbose:
      print(f" - Doing K-means chi squared")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist

    scaler = StandardScaler()
    columns = [col for col in self.sim_dataset.columns if col != "wt"]
    sim_scaled = scaler.fit_transform(self.sim_dataset.loc[:,columns])

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(sim_scaled)

    centroids = kmeans.cluster_centers_

    sim_distances = cdist(sim_scaled, centroids, metric='euclidean')
    sim_bins = np.argmin(sim_distances, axis=1)
    sim_histogram = np.array([np.sum(self.sim_dataset.loc[(sim_bins == i),"wt"]) for i in range(kmeans.n_clusters)])
    sim_histogram_uncert = np.sqrt(np.array([np.sum(self.sim_dataset.loc[(sim_bins == i),"wt"]**2) for i in range(kmeans.n_clusters)]))

    synth_scaled = scaler.transform(self.synth_dataset.loc[:, columns])
    synth_distances = cdist(synth_scaled, centroids, metric='euclidean')
    synth_bins = np.argmin(synth_distances, axis=1)
    synth_histogram = np.array([np.sum(self.synth_dataset.loc[(synth_bins == i),"wt"]) for i in range(kmeans.n_clusters)])
    synth_histogram_uncert = np.sqrt(np.array([np.sum(self.synth_dataset.loc[(synth_bins == i),"wt"]**2) for i in range(kmeans.n_clusters)]))

    non_zero_indices = np.where((synth_histogram > 0) & (sim_histogram > 0))[0]
    sim_histogram = sim_histogram[non_zero_indices]
    sim_histogram_uncert = sim_histogram_uncert[non_zero_indices]
    synth_histogram = synth_histogram[non_zero_indices]
    synth_histogram_uncert = synth_histogram_uncert[non_zero_indices]

    sum_sim = np.sum(sim_histogram)
    sum_synth = np.sum(synth_histogram)
    sim_histogram /= sum_sim
    sim_histogram_uncert /= sum_sim
    synth_histogram /= sum_synth
    synth_histogram_uncert /= sum_synth

    chi_squared = float(np.sum((synth_histogram-sim_histogram)**2/(synth_histogram_uncert**2 + sim_histogram_uncert**2)))
    dof = len(synth_histogram)
    chi_squared_per_dof = chi_squared / dof

    return chi_squared_per_dof


  def DoWassersteinSliced(self):

    if self.verbose:
      print(f" - Doing sliced wasserstein")

    n_features = len(self.columns)

    # Generate random projection vectors (normalized)
    random_directions = np.random.normal(size=(self.wasserstein_slices, n_features))
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)

    swd = 0.0
    for direction in random_directions:

      # Project data onto the random direction
      proj1 = np.dot(self.sim_dataset.loc[:,self.columns].to_numpy(), direction)
      proj2 = np.dot(self.synth_dataset.loc[:,self.columns].to_numpy(), direction)

      # Compute the Wasserstein distance in 1D
      swd += float(self._GetWasserstein(proj1, self.sim_dataset.loc[:,"wt"].to_numpy().flatten(), proj2, self.synth_dataset.loc[:,"wt"].to_numpy().flatten()))

      #proj1, proj1_wt = self._BinNegativeWeightedEvents(proj1.flatten(), self.sim_dataset.loc[:,"wt"].to_numpy().flatten())
      #swd += float(wasserstein_distance(proj1, proj2, u_weights=proj1_wt))

    # Average over the number of projections
    swd /= self.wasserstein_slices

    return float(swd)


  def DoWassersteinUnbinned(self):

    if self.verbose:
      print(f" - Doing unbinned wasserstein on X columns")

    wasserstein_unbinned = {}
    for col in self.columns:
      wasserstein_unbinned[col] =  float(self._GetWasserstein(self.sim_dataset.loc[:,col].to_numpy().flatten(), self.sim_dataset.loc[:,"wt"].to_numpy().flatten(), self.synth_dataset.loc[:,col].to_numpy().flatten(), self.synth_dataset.loc[:,"wt"].to_numpy().flatten()))

      #sim_no_neg, wt_no_neg = self._BinNegativeWeightedEvents(self.sim_dataset.loc[:,col].to_numpy().flatten(), self.sim_dataset.loc[:,"wt"].to_numpy().flatten())
      #wasserstein_unbinned[col] = float(wasserstein_distance(sim_no_neg, self.synth_dataset.loc[:,col], u_weights=wt_no_neg))

    return wasserstein_unbinned


  def MakeDatasets(self, only_synth=False, only_sim=False, seed=42):

    if not only_synth:

      if self.verbose:
        print(f" - Loading simulated dataset")

      # Get simulated data
      sim_dp = DataProcessor(
        self.sim_files,
        "parquet",
        wt_name = self.sim_wt_name,
        options = {
          "functions" : self.functions_to_apply,
          "selection": self.sim_selection
        }
      )

      if self.metrics != ["BDT Separation"]:
        self.sim_dataset = sim_dp.GetFull(method="sampled_dataset", sampling_fraction=self.sim_fraction)      
      if "BDT Separation" in self.metrics:
        self.sim_train, self.sim_test = sim_dp.GetFull(method="train_test_split", test_fraction=0.5, seed=seed)

    if not only_sim:

      if self.verbose:
        print(f" - Loading synthetic dataset")

      # Get simulated data
      synth_dp = DataProcessor(
        self.synth_files,
        "parquet",
        wt_name = self.synth_wt_name,
        options = {
          "functions" : self.functions_to_apply,
          "selection": self.synth_selection
        }
      )
      if self.metrics != ["BDT Separation"]:
        self.synth_dataset = synth_dp.GetFull("sampled_dataset", sampling_fraction=self.synth_fraction) 
      if "BDT Separation" in self.metrics:
        self.synth_train, self.synth_test = synth_dp.GetFull(method="train_test_split", test_fraction=0.5, seed=seed+1)


  def Run(self, seed=42, make_datasets=True):

    print("WARNING: MultiDim metrics involves loading a lot of data into memory. This option should preferably be run on a GPU. If you are struggling with memory usage, reduce the fraction of events.")

    # set seed
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    metrics = {}
    if make_datasets:
      self.MakeDatasets(seed=seed)

    if "BDT Separation" in self.metrics:
      auc, accuracy = self.DoBDTSeparation()
      metrics[f"bdt_auc"] = auc
      metrics[f"bdt_accuracy"] = accuracy
      metrics.update(self.signed_bdt_metrics)
    if "K-Means Chi Squared" in self.metrics:
      metrics[f"k_means_chi_squared"] = self.DoKMeansChiSquared()
    if "Wasserstein Sliced" in self.metrics:
      metrics[f"wasserstein_sliced"] = self.DoWassersteinSliced()
    if "Wasserstein Unbinned" in self.metrics:
      metrics[f"wasserstein_unbinned"] = self.DoWassersteinUnbinned()

    return metrics
