import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

from functools import partial
from sklearn.metrics import auc as roc_curve_auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor
from useful_functions import Resample

class MultiDimMetrics():

  def __init__(self, sim_files, synth_files, parameters, sim_fraction=1.0, synth_fraction=1.0):
    self.sim_files = sim_files
    self.synth_files = synth_files
    self.sim_fraction = sim_fraction
    self.synth_fraction = synth_fraction
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    self.verbose = True
    self.wasserstein_slices = 100
    self.resample_sim = False
    self.parameters = parameters

    self.sim_dataset = None
    self.synth_dataset = None
    self.sim_train = None
    self.sim_test = None
    self.synth_train = None
    self.synth_test = None
    self.metrics = []

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

    # Sort data and weights for both datasets
    idx1 = np.argsort(data1)
    sorted_data1, sorted_weights1 = data1[idx1], weights1[idx1] / np.sum(weights1)
    
    idx2 = np.argsort(data2)
    sorted_data2, sorted_weights2 = data2[idx2], weights2[idx2] / np.sum(weights2)

    # Calculate the cumulative weights (CDF values)
    cdf1 = np.cumsum(sorted_weights1)
    cdf2 = np.cumsum(sorted_weights2)

    # Align data and interpolate CDFs (ensure domains overlap properly)
    all_data = np.sort(np.unique(np.concatenate([sorted_data1, sorted_data2])))
    cdf1_interp = np.interp(all_data, sorted_data1, cdf1, left=0, right=1)
    cdf2_interp = np.interp(all_data, sorted_data2, cdf2, left=0, right=1)
    
    # Wasserstein distance: sum of weighted differences between the aligned CDFs
    return np.sum(np.abs(cdf1_interp - cdf2_interp) * np.diff(np.concatenate([[0], all_data])))

  def AddBDTSeparation(self):
    self.metrics.append("BDT Separation")

  def AddKMeansChiSquared(self):
    self.metrics.append("K-Means Chi Squared")

  def AddWassersteinSliced(self):
    self.metrics.append("Wasserstein Sliced")

  def AddWassersteinUnbinned(self):
    self.metrics.append("Wasserstein Unbinned")

  def DoBDTSeparation(self, remove_neg_weights=False, no_conditions=False):

    if self.verbose:
      print(f" - Doing BDT separation")

    import xgboost as xgb

    # Make training and testing datasets
    if no_conditions:
      train_columns = self.parameters["X_columns"]
    else:
      train_columns = self.parameters["X_columns"] + self.parameters["Y_columns"]

    # Make training and testing datasets
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
      self.sim_train.loc[:, "y"] = 0.0
      self.synth_train.loc[:, "y"] = 1.0
      self.sim_test.loc[:, "y"] = 0.0
      self.synth_test.loc[:, "y"] = 1.0
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
      Xy_sim_train, wt_sim_train = Resample(Xy_sim_train, wt_sim_train, n_samples=len(Xy_sim_train))
      Xy_sim_test, wt_sim_test = Resample(Xy_sim_test, wt_sim_test, n_samples=len(Xy_sim_test))
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
    #youden_index = np.argmax(tpr - fpr)
    #optimal_threshold = thresholds[youden_index]
    #y_pred_train = (y_prob_train >= optimal_threshold).astype(int)
    y_pred_train = (y_prob_train >= 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, y_pred_train, sample_weight=wt_train)

    # Get test auc
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test, sample_weight=wt_test)
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    test_auc = roc_curve_auc(fpr, tpr)

    # Get test accuracy
    #youden_index = np.argmax(tpr - fpr)
    #optimal_threshold = thresholds[youden_index]
    #print(optimal_threshold)
    #y_pred_test = (y_prob_test >= optimal_threshold).astype(int)
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_test, sample_weight=wt_test)

    if self.verbose:
      print(f" - Train AUC: {train_auc}")
      print(f" - Test AUC: {test_auc}")
      print(f" - Train Accuracy: {train_accuracy}")
      print(f" - Test Accuracy: {test_accuracy}")

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

    n_features = len(self.parameters["X_columns"]+self.parameters["Y_columns"])

    # Generate random projection vectors (normalized)
    random_directions = np.random.normal(size=(self.wasserstein_slices, n_features))
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)

    swd = 0.0
    for direction in random_directions:

      # Project data onto the random direction
      proj1 = np.dot(self.sim_dataset.loc[:,self.parameters["X_columns"]+self.parameters["Y_columns"]].to_numpy(), direction)
      proj2 = np.dot(self.synth_dataset.loc[:,self.parameters["X_columns"]+self.parameters["Y_columns"]].to_numpy(), direction)

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
    for col in self.parameters["X_columns"]:
      wasserstein_unbinned[col] =  float(self._GetWasserstein(self.sim_dataset.loc[:,col].to_numpy().flatten(), self.sim_dataset.loc[:,"wt"].to_numpy().flatten(), self.synth_dataset.loc[:,col].to_numpy().flatten(), self.synth_dataset.loc[:,"wt"].to_numpy().flatten()))

      #sim_no_neg, wt_no_neg = self._BinNegativeWeightedEvents(self.sim_dataset.loc[:,col].to_numpy().flatten(), self.sim_dataset.loc[:,"wt"].to_numpy().flatten())
      #wasserstein_unbinned[col] = float(wasserstein_distance(sim_no_neg, self.synth_dataset.loc[:,col], u_weights=wt_no_neg))

    return wasserstein_unbinned


  def MakeDatasets(self, only_synth=False, only_sim=False):

    if not only_synth:

      if self.verbose:
        print(f" - Loading simulated dataset")

      # Get simulated data
      sim_dp = DataProcessor(
        self.sim_files,
        "parquet",
        wt_name = "wt",
        options = {}
      )
      if self.metrics != ["BDT Separation"]:
        self.sim_dataset = sim_dp.GetFull(method="sampled_dataset", sampling_fraction=self.sim_fraction)      
      if "BDT Separation" in self.metrics: # This ensures that duplicated data from different validation points are split in the same way
        self.sim_train, self.sim_test = sim_dp.GetFull(method="train_test_split", test_fraction=0.5)

    if not only_sim:

      if self.verbose:
        print(f" - Loading synthetic dataset")

      # Get simulated data
      synth_dp = DataProcessor(
        self.synth_files,
        "parquet",
        wt_name = "wt",
        options = {}
      )
      if self.metrics != ["BDT Separation"]:
        self.synth_dataset = synth_dp.GetFull("sampled_dataset", sampling_fraction=self.synth_fraction) 
      if "BDT Separation" in self.metrics: # This ensures that duplicated data from different validation points are split in the same way
        self.synth_train, self.synth_test = synth_dp.GetFull(method="train_test_split", test_fraction=0.5)
      

  def Run(self, seed=42, make_datasets=True):

    print("WARNING: MultiDim metrics involves loading a lot of data into memory. This option should preferably be run on a GPU. If you are struggling with memory usage, reduce the fraction of events.")

    # set seed
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    metrics = {}
    if make_datasets:
      self.MakeDatasets()

    if "BDT Separation" in self.metrics:
      auc, accuracy = self.DoBDTSeparation()
      metrics[f"bdt_auc"] = auc
      metrics[f"bdt_accuracy"] = accuracy
    if "K-Means Chi Squared" in self.metrics:
      metrics[f"k_means_chi_squared"] = self.DoKMeansChiSquared()
    if "Wasserstein Sliced" in self.metrics:
      metrics[f"wasserstein_sliced"] = self.DoWassersteinSliced()
    if "Wasserstein Unbinned" in self.metrics:
      metrics[f"wasserstein_unbinned"] = self.DoWassersteinUnbinned()

    return metrics
