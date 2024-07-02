import yaml
import copy

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from math import gamma
from scipy.stats import vonmises
from scipy.stats import beta

from useful_functions import MakeDirectories

class Dim5():

  def __init__(self, file_name=None): 
    """
    A class used to simulate and handle a 5-dimensional dataset.

    Parameters
    ----------
    file_name : str, optional
        The name of the file to save or load the dataset (default is None).
    """

    self.name = "Dim5"
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
    self.signal_resolution = 0.01
    self.chi = 3.5
    self.exponential_factor = 20
    self.kappa =  4.0
    self.alpha = 0.5
    self.beta = 0.5

    self.signal_yield = 1000.0
    self.train_test_val_split = "0.4:0.3:0.3"
    self.array_size = int(3e6)
    self.dir_name = f"data/Benchmark_{self.name}/Inputs"
    

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
      "files" : {"Signal" : f"{self.dir_name}/Signal.parquet"},
      "variables" : [
        "X1",
        "X2",
        "X3",
        "X4",
        "X5"
      ],
      "pois" : ["Y1"],
      "nuisances" : [],
      "preprocess" : {
        "standardise" : "all",
        "train_test_val_split" : self.train_test_val_split,
        "equalise_y_wts" : True,
        "train_test_y_vals" : {"Y1" : self.train_test_y_vals},
        "validation_y_vals" : {"Y1" : self.validation_y_vals}
      },
      "inference" : {},
      "validation" : {},
      "data_file" : f"{self.dir_name}/{self.name}_data.parquet"
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
    Y_df = pd.DataFrame(
      {
        "Y1" : np.random.choice(self.true_values, size=self.array_size),
      }
    )
    self.file_name = "Signal"

    df = self.Sample(Y_df, n_events=len(Y_df))
    df.loc[:, "Y1"] = Y_df
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

  def Probability(self, X, Y, return_log_prob=True):
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

    if self.file_name == "Signal":

      # Get X1 pdf
      std_dev = float(self.signal_resolution * Y.loc[0,"Y1"])
      mean = float(Y.loc[0,"Y1"])
      gaussian_pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(X.loc[:,"X1"] - mean)**2 / (2 * std_dev**2))

      # Get X2 pdf
      k_chi_1 = self.chi + (Y.loc[0,"Y1"] - 173.0) * 0.1 
      chi_pdf = ((1/2)**(k_chi_1/2)) / gamma(k_chi_1/2) * (X.loc[:,"X2"]**(k_chi_1/2 - 1)) * np.exp(-X.loc[:,"X2"]/2)

      # Get X3 pdf
      beta_val = self.exponential_factor + Y.loc[0,"Y1"] * self.exponential_factor / ((Y.loc[0,"Y1"] - 160.0)**2)
      exponential_pdf = (1/beta_val)*np.exp(-X.loc[:,"X3"]/beta_val)

      # Get X4 pdf
      alpha = self.alpha + Y.loc[0,"Y1"] * 0.01
      beta_val = self.beta * (Y.loc[0,"Y1"] - 165.0) * 0.1
      beta_pdf = beta.pdf(X.loc[:,"X4"], alpha, beta_val)

      # Get X5 pdf
      lambda_w, k_w = Y.loc[0,"Y1"] * 0.05 , (Y.loc[0,"Y1"] - 160.0) ** 2.0
      weibull_pdf = (k_w / lambda_w) * (X.loc[:,"X5"] / lambda_w)**(k_w - 1) * np.exp(-(X.loc[:,"X5"] / lambda_w)**k_w)

      pdf = gaussian_pdf * chi_pdf * exponential_pdf * beta_pdf * weibull_pdf

    # Return correct value
    if return_log_prob:
      return np.log(pdf.to_numpy()).reshape(-1,1)
    else:
      return pdf.to_numpy().reshape(-1,1)

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

    if self.file_name == "Signal":

      # Set up Y correctly
      if len(Y) == 1:
        Y = pd.DataFrame(np.tile(Y.to_numpy().flatten(), (n_events, 1)), columns=Y.columns, dtype=np.float64)

      # Define beta function
      def beta_rvs(Y):
        alpha = self.alpha + Y * 0.01
        beta_val = self.beta * (Y - 165.0) * 0.1
        beta_rvs_val = beta.rvs(alpha, beta_val)
        if beta_rvs_val == 1.0: # To avoid breaking the pdf
          beta_rvs_val = beta_rvs(Y)        
        return beta_rvs_val

      # Define weibull function
      def weibull(Y):
        lambda_w, k_w = Y * 0.05 , (Y - 160.0) ** 2.0
        return np.random.weibull(k_w) * lambda_w

      # Make dataframe
      df = pd.DataFrame(
        {
          "X1" : np.random.normal(Y.loc[:,"Y1"].to_numpy(), self.signal_resolution*Y.loc[:,"Y1"].to_numpy()),
          "X2" : np.random.chisquare(df=(self.chi + (Y.loc[:,"Y1"].to_numpy() - 173.0) * 0.1), size=len(Y)),
          "X3" : np.random.exponential(self.exponential_factor + Y.loc[:,"Y1"] * self.exponential_factor / ((Y.loc[:,"Y1"] - 160.0)**2)),
          "X4" : Y["Y1"].apply(beta_rvs),
          "X5" : Y["Y1"].apply(weibull),
        },
        dtype=np.float64
      )

    return df