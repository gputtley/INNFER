import copy
import os
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from useful_functions import MakeDirectories

data_dir = str(os.getenv("DATA_DIR"))
plots_dir = str(os.getenv("PLOTS_DIR"))
models_dir = str(os.getenv("MODELS_DIR"))

class Dim1GaussianWithExpBkg():

  def __init__(self, file_name=None): 
    """
    A class used to simulate and handle a 1-dimensional Gaussian dataset with
    an exponentially falling background.

    Parameters
    ----------
    file_name : str, optional
        The name of the file to save or load the dataset (default is None).
    """

    self.name = "Dim1GaussianWithExpBkg"
    self.file_name = file_name
    self.true_values = [
      166.0,167.0,168.0,169.0,170.0,
      170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
      176.0,177.0,178.0,179.0,180.0,
    ]
    self.train_test_y_vals = [
      166.0,167.0,168.0,169.0,170.0,
      171.0,172.0,173.0,174.0,175.0,
      176.0,177.0,178.0,179.0,180.0,           
    ]
    self.validation_y_vals = [
      171.0,171.5,172.0,172.5,173.0,
      173.5,174.0,174.5,175.0,      
    ]
    self.data_value = 172.345
    self.default_value = 172.5
    self.signal_resolution = 0.01
    self.signal_fraction = 0.3
    self.background_ranges = [160.0,185.0]
    self.background_lambda = 0.1
    self.background_constant = 160.0
    self.signal_yield = 1000.0
    self.train_test_val_split = "0.4:0.3:0.3"
    self.array_size = int(3e6)
    self.saved_parameters = {}
    self.dir_name = f"{data_dir}/Benchmark_{self.name}/Inputs"    


  def Load(self, name=None):
    pass


  def MakeConfig(self, return_cfg=False):
    """
    Creates a configuration dictionary for the dataset and saves it as a YAML file.

    Parameters
    ----------
    return_cfg : bool, optional
        If True, the configuration dictionary is returned (default is False).

    Returns
    -------
    dict
        The configuration dictionary if return_cfg is True.
    """

    cfg = {
      "name" : f"Benchmark_{self.name}",
      "variables" : ["X1"],
      "pois" : ["Y1"],
      "nuisances" : [],
      "data_file" : f"{self.dir_name}/{self.name}_data.parquet",
      "inference" : {
        "nuisance_constraints" : [],
        "rate_parameters" : [],
        "lnN" : {},
      },
      "default_values" : {
        "Y1" : self.default_value,
      },
      "models" : {
        "GaussianWithExpBkg" : {
          "density_models" : [
            {
              "parameters" : ["Y1"],
              "file" : "base",
              "shifts" : {}
            },
          ],
          "yields" : [{"file" : "base"}]
        }
      },
      "validation": {
        "loop" : [{"Y1" : i} for i in self.validation_y_vals],
        "files" : {"GaussianWithExpBkg" : [{"file": "base"}]}
      },
      "preprocess" : {
        "train_test_val_split" : self.train_test_val_split,
        "drop_from_training" : {"GaussianWithExpBkg" : {"Y1" : [k for k in self.true_values if k not in self.train_test_y_vals]}},
      },
      "files" : {
        "base" : {
          "inputs" : [f"{self.dir_name}/GaussianWithExpBkg.parquet"],
          "weight" : "wt",
          "parameters" : ["Y1"],
        },
      }
    }


    if return_cfg:
      return cfg

    with open(f"configs/run/Benchmark_{self.name}.yaml", 'w') as file:
      yaml.dump(cfg, file)


  def MakeDataset(self):
    """
    Creates a simulated dataset and saves it as a Parquet file.
    """

    # Make directory
    MakeDirectories(self.dir_name)

    # Make input simulated dataframe
    df = pd.DataFrame(
      {
        "Y1" : np.random.choice(self.true_values, size=self.array_size),
      }
    )
    self.file_name = "GaussianWithExpBkg"
    df.loc[:, "X1"] = self.Sample(df, n_events=len(df))
    df.loc[:, "wt"] = np.ones(len(df))

    # Rescale weights so all are equivalent
    for true_value in self.true_values:
      df.loc[(df.loc[:,"Y1"] == true_value), "wt"] *= float(self.signal_yield) / float(np.sum(df.loc[(df.loc[:,"Y1"] == true_value), "wt"].to_numpy(), dtype=np.float128))

    # Write to file
    table = pa.Table.from_pandas(df)
    parquet_file_path = f"{self.dir_name}/{self.file_name}.parquet"
    pq.write_table(table, parquet_file_path)

    # Make toy data dataset
    data = self.Sample(pd.DataFrame({"Y1" : [self.data_value]}, dtype=np.float128), int(self.signal_yield))
    data_table = pa.Table.from_pandas(data)
    data_parquet_file_path = f"{self.dir_name}/{self.name}_data.parquet"
    pq.write_table(data_table, data_parquet_file_path)


  def Probability(self, X, Y, return_log_prob=True, order=0, column_1=None, column_2=None):
    """
    Computes the probability density function for a Gaussian distribution.

    Parameters
    ----------
    X : pandas.DataFrame
        The input dataframe containing X values.
    Y : pandas.DataFrame
        The input dataframe containing Y values.
    return_log_prob : bool, optional
        If True, the log probability is returned (default is True).

    Returns
    -------
    numpy.ndarray
        The (log) probability values.
    """

    if order != 0 and order != [0]:
      raise ValueError("Analytical derivatives are not setup for the benchmark")

    if Y.loc[0,"Y1"] < min(self.train_test_y_vals) or Y.loc[0,"Y1"] > max(self.train_test_y_vals):
      if order == 0:
        return np.full((len(X), 1), np.nan)
      else:
        return [np.full((len(X), 1), np.nan)]
      
    if self.file_name == "GaussianWithExpBkg":

      # Get gaussian PDF
      std_dev = float(self.signal_resolution * Y.loc[0,"Y1"])
      mean = float(Y.loc[0,"Y1"])
      sig_pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(X.loc[:,"X1"] - mean)**2 / (2 * std_dev**2))

      # Get exp background pdf
      def BkgPDFUnNorm(x):
        return self.background_lambda*np.exp(-self.background_lambda*(x-self.background_constant))
      if "background_normalisation" not in self.saved_parameters:
        int_space = np.linspace(self.background_ranges[0],self.background_ranges[1],num=100)
        bkg_pdf_unnorm = BkgPDFUnNorm(int_space)
        self.saved_parameters["background_normalisation"] = np.sum(bkg_pdf_unnorm) * (int_space[1]-int_space[0])
      bkg_pdf = BkgPDFUnNorm(X.loc[:,"X1"])/self.saved_parameters["background_normalisation"]

      pdf = (self.signal_fraction*sig_pdf) + ((1-self.signal_fraction)*bkg_pdf)

    # Return correct value
    if return_log_prob:
      if order == 0:
        return np.log(pdf.to_numpy()).reshape(-1,1)
      else:
        return [np.log(pdf.to_numpy()).reshape(-1,1)]
    else:
      if order == 0:
        return pdf.to_numpy().reshape(-1,1)
      else:
        return [pdf.to_numpy().reshape(-1,1)]


  def Sample(self, Y, n_events):
    """
    Samples values from a Gaussian distribution.

    Parameters
    ----------
    Y : pandas.DataFrame
        The input dataframe containing Y values.
    n_events : int
        The number of events to sample.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing sampled X values.
    """

    if self.file_name == "GaussianWithExpBkg":

      # Set up Y correctly
      if len(Y) == 1:
        Y = pd.DataFrame(np.tile(Y.to_numpy().flatten(), (n_events, 1)), columns=Y.columns, dtype=np.float64)

      signal_entries = int(round(n_events*self.signal_fraction))

      signal_indices = np.random.choice(Y.loc[:,"Y1"].to_numpy().shape[0], size=signal_entries, replace=False)
      bkg_indices = np.setdiff1d(np.arange(Y.loc[:,"Y1"].to_numpy().shape[0]), signal_indices)

      X_signal = np.random.normal(Y.loc[:,"Y1"].to_numpy()[signal_indices], self.signal_resolution*Y.loc[:,"Y1"].to_numpy()[signal_indices])
      bkg_entries = len(bkg_indices)
      X_bkg = np.zeros(bkg_entries)
      for ind in range(bkg_entries):
        x = np.random.exponential(scale=1/self.background_lambda) + self.background_constant
        while x < self.background_ranges[0] or x > self.background_ranges[1]:
          x = np.random.exponential(scale=1/self.background_lambda) + self.background_constant
        X_bkg[ind] = x

      X = np.vstack((X_signal.reshape(-1,1),X_bkg.reshape(-1,1)))
      indices = np.concatenate((signal_indices,bkg_indices))
      sorted_order = np.argsort(indices)
      X = X[sorted_order]

      # Make dataframe
      df = pd.DataFrame(
        {
          "X1" : X.flatten()
        },
        dtype=np.float64
      )

    return df