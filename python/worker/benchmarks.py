import yaml
import copy
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import i0  
from scipy.stats import vonmises
from scipy.stats import beta
from scipy.stats import weibull_min
from math import factorial, gamma

class Benchmarks():
  """
  A class for generating synthetic datasets and configurations for benchmarks.
  """
  def __init__(self, name="Gaussian", parameters={}):
    """
    Initialize the Benchmarks class.

    Args:
        name (str): Name of the benchmark.
        parameters (dict): Additional parameters for the benchmark.
    """
    self.name = name

    self.model_parameters = {
      "Gaussian": {
        "signal_resolution" : 0.01,
        "true_masses" : [
          166.0,167.0,168.0,169.0,170.0,
          170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          176.0,177.0,178.0,179.0,180.0,
        ],
        "toy_signal_events" : 1000.0,
      },
      "GaussianWithExpBkg": {
        "signal_resolution" : 0.01,
        "signal_fraction" : 0.3,
        "background_ranges" : [160.0,185.0],
        "background_lambda" : 0.1,
        "background_constant" : 160.0,
        "true_masses" : [
          166.0,167.0,168.0,169.0,170.0,
          170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          176.0,177.0,178.0,179.0,180.0,
        ],
        "toy_signal_events" : 1000.0,
      },
      "GaussianWithExpBkgVaryingYield": {
        "signal_resolution" : 0.01,
        "background_ranges" : [160.0,185.0],
        "background_lambda" : 0.1,
        "background_constant" : 160.0,
        "true_masses" : [
          166.0,167.0,168.0,169.0,170.0,
          170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          176.0,177.0,178.0,179.0,180.0,
        ],    
        "toy_signal_events" : 1000.0,
        "toy_background_events" : 1000.0,
      },
      "2D": {
        "signal_resolution" : 0.01,
        "chi_1" : 3.5,
        "true_masses" : [
          166.0,167.0,168.0,169.0,170.0,
          170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          176.0,177.0,178.0,179.0,180.0,
        ],    
        "toy_signal_events" : 1000.0,
       },
      "5D": { 
        "signal_resolution" : 0.01,
        "exponential_factor_1" : 20,
        "exponential_factor_2" : 5,
        "exponential_factor_3" : 10,
        "kappa": 4.0,
        "true_masses" : [
          166.0,167.0,168.0,169.0,170.0,
          170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          176.0,177.0,178.0,179.0,180.0,
        ],    
        "toy_signal_events" : 1000.0,
      },
      "5D+D": {
        "signal_resolution" : 0.01,
        "chi" : 5.0,
        "alpha" : 0.5,
        "beta" : 2.0,
        "exponential_factor" : 20,
        "true_masses" : [
          166.0,167.0,168.0,169.0,170.0,
          170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          176.0,177.0,178.0,179.0,180.0,
        ],    
        "toy_signal_events" : 1000.0,
      },
      "12D": {
        "signal_resolution" : 0.01,
        "signal_resolution_1" : 0.02,
        "exponential_factor_1" : 20,
        "exponential_factor_2" : 5,
        "exponential_factor_3" : 10,
        "kappa": 4.0,
        "kappa_2" : 2.0,
        "chi_1" : 3.0,
        "chi_2" : 5.0,
        "alpha_7" : 0.5,
        "alpha_8" : 2.0,
        "beta_7" : 0.5,
        "beta_8" : 5.0,
        "true_masses" : [
          166.0,167.0,168.0,169.0,170.0,
          170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          176.0,177.0,178.0,179.0,180.0,
        ],    
        "toy_signal_events" : 1000.0,
      }
    }

    self.saved_parameters = {}
    self.array_size = int(3e6)

    for k, v in parameters.items():
      self.model_parameters[name][k] = v


  def GetPDF(self, file_name):
    """
    Get a PDF function based on the specified benchmark and file name.

    Args:
        file_name (str): Name of the file.

    Returns:
        function: PDF function for the specified benchmark and file.
    """
    def return_pdf(X, Y):
      return self.PDF(X,Y,file_name)
    return return_pdf

  def PDF(self, X, Y, file_name):
    """
    Calculate the probability density function (PDF) for a given X, Y, and file name.

    Args:
        X (array-like): Values of the X variable.
        Y (array-like): Values of the Y variable.
        file_name (str): Name of the file.

    Returns:
        array-like: PDF values.
    """
    if isinstance(X, float) or isinstance(X, int): X = np.array([X])
    if isinstance(Y, float) or isinstance(Y, int): Y = np.array([Y])

    if file_name == "Gaussian":

      std_dev = self.model_parameters[self.name]["signal_resolution"] * Y[0]
      mean = Y[0]
      pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(X[0] - mean)**2 / (2 * std_dev**2))
      return pdf
    
    elif file_name == "GaussianWithExpBkg":

      std_dev = self.model_parameters[self.name]["signal_resolution"] * Y[0]
      mean = Y[0]
      sig_pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(X[0] - mean)**2 / (2 * std_dev**2))
      def BkgPDFUnNorm(x):
        return self.model_parameters[self.name]["background_lambda"]*np.exp(-self.model_parameters[self.name]["background_lambda"]*(x-self.model_parameters[self.name]["background_constant"]))
      if "background_normalisation" not in self.saved_parameters:
        int_space = np.linspace(self.model_parameters[self.name]["background_ranges"][0],self.model_parameters[self.name]["background_ranges"][1],num=100)
        bkg_pdf_unnorm = BkgPDFUnNorm(int_space)
        self.saved_parameters["background_normalisation"] = np.sum(bkg_pdf_unnorm) * (int_space[1]-int_space[0])
      bkg_pdf = BkgPDFUnNorm(X[0])/self.saved_parameters["background_normalisation"]
      return (self.model_parameters[self.name]["signal_fraction"]*sig_pdf) + ((1-self.model_parameters[self.name]["signal_fraction"])*bkg_pdf)

    elif file_name == "ExpBkg":

      def BkgPDFUnNorm(x):
        return self.model_parameters[self.name]["background_lambda"]*np.exp(-self.model_parameters[self.name]["background_lambda"]*(x-self.model_parameters[self.name]["background_constant"]))
      if "background_normalisation" not in self.saved_parameters:
        int_space = np.linspace(self.model_parameters[self.name]["background_ranges"][0],self.model_parameters[self.name]["background_ranges"][1],num=100)
        bkg_pdf_unnorm = BkgPDFUnNorm(int_space)
        self.saved_parameters["background_normalisation"] = np.sum(bkg_pdf_unnorm) * (int_space[1]-int_space[0])
      bkg_pdf = BkgPDFUnNorm(X[0])/self.saved_parameters["background_normalisation"]     
      return bkg_pdf
    
    elif self.name == "GaussianWithExpBkgVaryingYield" and file_name == "combined":

      std_dev = self.model_parameters[self.name]["signal_resolution"] * Y[1]
      mean = Y[1]
      sig_pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(X[0] - mean)**2 / (2 * std_dev**2))
      def BkgPDFUnNorm(x):
        return self.model_parameters[self.name]["background_lambda"]*np.exp(-self.model_parameters[self.name]["background_lambda"]*(x-self.model_parameters[self.name]["background_constant"]))
      if "background_normalisation" not in self.saved_parameters:
        int_space = np.linspace(self.model_parameters[self.name]["background_ranges"][0],self.model_parameters[self.name]["background_ranges"][1],num=100)
        bkg_pdf_unnorm = BkgPDFUnNorm(int_space)
        self.saved_parameters["background_normalisation"] = np.sum(bkg_pdf_unnorm) * (int_space[1]-int_space[0])
      bkg_pdf = BkgPDFUnNorm(X[0])/self.saved_parameters["background_normalisation"]
      #print(X,Y,((Y[0]*sig_pdf) + bkg_pdf)/(Y[0]+1))
      return ((Y[0]*sig_pdf) + bkg_pdf)/(Y[0]+1)

    elif file_name == "2D":

      std_dev = self.model_parameters[self.name]["signal_resolution"] * Y[0]
      mean = Y[0]
      gaussian_pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) *  np.exp(-(X[0] - mean)**2 / (2 * std_dev**2))

      k_chi_1 = self.model_parameters[self.name]["chi_1"] + (Y[0] - 173.0) * 0.1 
      chi_pdf_1 = ((1/2)**(k_chi_1/2)) / gamma(k_chi_1/2) * (X[1]**(k_chi_1/2 - 1)) * np.exp(-X[1]/2)

      return gaussian_pdf * chi_pdf_1

    elif file_name == "5D":

      # gaussian
      std_dev_g = self.model_parameters[self.name]["signal_resolution"] * Y[0]
      mean_g = Y[0]
      gaussian_pdf = (1 / (std_dev_g * np.sqrt(2 * np.pi))) *  np.exp(-(X[0] - mean_g)**2 / (2 * std_dev_g**2))

      # expoential for 3 features
      beta_1 = Y[0] / self.model_parameters[self.name]["exponential_factor_1"]
      beta_2 = Y[0] / self.model_parameters[self.name]["exponential_factor_2"]
      beta_3 = Y[0] / self.model_parameters[self.name]["exponential_factor_3"]
      exponential_pdf_1 = (1/beta_1)*np.exp(-X[1]/beta_1) # add a linear shift
      exponential_pdf_2 = (1/beta_2)*np.exp(-X[2]/beta_2) 
      exponential_pdf_3 = (1/beta_3)*np.exp(-X[3]/beta_3)

      # von mises 
      mu_v = 0.0  # Assuming mean is related to Y[0], adjust as necessary
      kappa_v = (1/ (self.model_parameters[self.name]["kappa"] * Y[0] * 0.1))**10 * (10**20)
      von_mises_pdf = np.exp(kappa_v * np.cos(X[4] - mu_v)) / (2 * np.pi * i0(kappa_v))
      
      return gaussian_pdf * exponential_pdf_1 * exponential_pdf_2 * exponential_pdf_3 * von_mises_pdf

    elif file_name == "5D+D":

      # gaussian
      std = self.model_parameters[self.name]["signal_resolution"] * Y[0]
      mean = Y[0]
      gaussian_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(X[0] - mean)**2 / (2 * std**2))

      # chi
      k_chi = self.model_parameters[self.name]["chi"] + (Y[0] - 173.0) * 0.1
      chi_pdf = ((1/2)**(k_chi/2)) / gamma(k_chi/2) * (X[1]**(k_chi/2 - 1)) * np.exp(-X[1]/2)

      # exponential
      beta_e = self.model_parameters[self.name]["exponential_factor"] + Y[0] * self.model_parameters[self.name]["exponential_factor"] / ((Y[0] - 160.0)**2)
      exponential_pdf = (1/beta_e)*np.exp(-X[2]/beta_e)

      # beta
      alpha_1 = self.model_parameters[self.name]["alpha"] + Y[0] * 0.02
      beta_1 = self.model_parameters[self.name]["beta"] * (Y[0] - 160.0) * 0.1
      beta_pdf = beta.pdf(X[3], alpha_1, beta_1)

      # weibull
      lambda_w, k_w = Y[0] * 0.05, (Y[0] - 160.0) ** 2.0
      weibull_pdf = (k_w / lambda_w) * (X[4] / lambda_w)**(k_w - 1) * np.exp(-(X[4] / lambda_w)**k_w)

      # discrete
      def discrete_pdf_func(x):
          probability_distribution = {
              0: 0.4,
              1: 0.2,
              2: 0.1,
              3: 0.1,
              4: 0.05,
              5: 0.05,
              6: 0.05,
              7: 0.05
          }
          return probability_distribution.get(x, 0)  # Returns 0 if x is not in the distribution

      discrete_pdf = discrete_pdf_func(X[5])
      
      return gaussian_pdf * chi_pdf * exponential_pdf * beta_pdf * weibull_pdf * discrete_pdf

    elif file_name == "12D":

      # gaussian
      std_dev_g = self.model_parameters[self.name]["signal_resolution"] * Y[0]
      mean_g = Y[0]
      gaussian_pdf = (1 / (std_dev_g * np.sqrt(2 * np.pi))) *  np.exp(-(X[0] - mean_g)**2 / (2 * std_dev_g**2))

      # expoential for 3 features
      beta_1 = self.model_parameters[self.name]["exponential_factor_1"] + Y[0] * self.model_parameters[self.name]["exponential_factor_1"] / ((Y[0] - 160.0)**2)
      beta_2 = self.model_parameters[self.name]["exponential_factor_2"] + Y[0] * self.model_parameters[self.name]["exponential_factor_2"] / ((Y[0] - 160.0)**2)
      beta_3 = self.model_parameters[self.name]["exponential_factor_3"] + Y[0] * self.model_parameters[self.name]["exponential_factor_3"] / ((Y[0] - 160.0)**2)
      exponential_pdf_1 = (1/beta_1)*np.exp(-X[1]/beta_1)
      exponential_pdf_2 = (1/beta_2)*np.exp(-X[2]/beta_2)
      exponential_pdf_3 = (1/beta_3)*np.exp(-X[3]/beta_3)

      # von mises 
      mu_v = 0.0  # Assuming mean is related to Y[0], adjust as necessary
      kappa_v = (1/ (self.model_parameters[self.name]["kappa"] * Y[0] * 0.1))**10 * (10**15)
      von_mises_pdf = np.exp(kappa_v * np.cos(X[4] - mu_v)) / (2 * np.pi * i0(kappa_v))
      
      kappa_v2 = (1/ (self.model_parameters[self.name]["kappa_2"] * Y[0] * 0.1))**10 * (10**10)
      von_mises_pdf_2 = np.exp(kappa_v2 * np.cos(X[5] - mu_v)) / (2 * np.pi * i0(kappa_v2))
    
      # the beta distribution for the 7,8 -th feature:
      alpha_7 = self.model_parameters[self.name]["alpha_7"] + Y[0] * 0.02
      beta_7 = self.model_parameters[self.name]["beta_7"] * (Y[0] - 160.0) * 0.1
      pdf_X7 = beta.pdf(X[6], alpha_7, beta_7)
      alpha_8 = self.model_parameters[self.name]["alpha_8"] + Y[0] * 0.01
      beta_8 = self.model_parameters[self.name]["beta_8"] * (Y[0] - 165.0) * 0.1
      pdf_X8 = beta.pdf(X[7], alpha_8, beta_8)

      # Chi-square distribution for the 9,10-th feature
      k_chi_1 = self.model_parameters[self.name]["chi_1"] + (Y[0] - 173.0) * 0.1
      k_chi_2 = self.model_parameters[self.name]["chi_2"] + (Y[0] - 173.0) * 0.1 
      chi_pdf_1 = ((1/2)**(k_chi_1/2)) / gamma(k_chi_1/2) * (X[8]**(k_chi_1/2 - 1)) * np.exp(-X[8]/2)
      chi_pdf_2 = ((1/2)**(k_chi_2/2)) / gamma(k_chi_2/2) * (X[9]**(k_chi_2/2 - 1)) * np.exp(-X[9]/2)

      # weibull distribution for the 11-th feature
      lambda_w, k_w = Y[0] * 0.05 , (Y[0] - 160.0) ** 2.0
      weibull_pdf = (k_w / lambda_w) * (X[10] / lambda_w)**(k_w - 1) * np.exp(-(X[10] / lambda_w)**k_w)

      # another guassian distribution for the 12-th feature
      std_dev_1 = self.model_parameters[self.name]["signal_resolution_1"] * Y[0]
      gaussian_pdf_1 = (1 / (std_dev_1 * np.sqrt(2 * np.pi))) *  np.exp(-(X[11] - mean_g)**2 / (2 * std_dev_1**2))

      PDF = gaussian_pdf * exponential_pdf_1 * exponential_pdf_2 * exponential_pdf_3 * von_mises_pdf * von_mises_pdf_2 * chi_pdf_1 * chi_pdf_2 * weibull_pdf * gaussian_pdf_1  * pdf_X7 * pdf_X8
      return  PDF
    
  def MakeDataset(self):
    """
    Generate synthetic datasets based on the benchmark type.
    """
    if self.name == "Gaussian":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      X = np.random.normal(Y, self.model_parameters[self.name]["signal_resolution"]*Y)
      df = pd.DataFrame(
        {
          "reconstructed_mass" : X, 
          "true_mass" : Y, 
          "wt" : (len(self.model_parameters[self.name]["true_masses"])*float(self.model_parameters[self.name]["toy_signal_events"])/float(self.array_size))*np.ones(int(self.array_size))
        },
        dtype=np.float64
      )

      # Rescale weights so all are equivalent
      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"true_mass"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(np.sum(df.loc[(df.loc[:,"true_mass"] == mass), "wt"]))

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}.parquet"
      pq.write_table(table, parquet_file_path)

    elif self.name == "GaussianWithExpBkg":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      signal_entries = int(round(self.array_size*self.model_parameters[self.name]["signal_fraction"]))
      X_signal = np.random.normal(Y[:signal_entries], self.model_parameters[self.name]["signal_resolution"]*Y[:signal_entries])
      bkg_entries = self.array_size - signal_entries
      X_bkg = np.zeros(bkg_entries)
      for ind in range(bkg_entries):
        x = np.random.exponential(scale=1/self.model_parameters[self.name]["background_lambda"]) + self.model_parameters[self.name]["background_constant"]
        while x < self.model_parameters[self.name]["background_ranges"][0] or x > self.model_parameters[self.name]["background_ranges"][1]:
          x = np.random.exponential(scale=1/self.model_parameters[self.name]["background_lambda"]) + self.model_parameters[self.name]["background_constant"]
        X_bkg[ind] = x
      X = np.vstack((X_signal.reshape(-1,1),X_bkg.reshape(-1,1)))
      df = pd.DataFrame(
        {
          "reconstructed_mass" : X.flatten(), 
          "true_mass" : Y.flatten(), 
          "wt" : (len(self.model_parameters[self.name]["true_masses"])*float(self.model_parameters[self.name]["toy_signal_events"])/float(signal_entries))*np.ones(int(self.array_size))
        },
        dtype=np.float64
      )
      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"true_mass"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(self.model_parameters[self.name]["signal_fraction"] * np.sum(df.loc[(df.loc[:,"true_mass"] == mass), "wt"]))

      df = df.sample(frac=1, random_state=42)
      df = df.reset_index(drop=True)

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}.parquet"
      pq.write_table(table, parquet_file_path)

    elif self.name == "GaussianWithExpBkgVaryingYield":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      X = np.random.normal(Y, self.model_parameters[self.name]["signal_resolution"]*Y)
      df = pd.DataFrame(
        {
          "reconstructed_mass" : X, 
          "true_mass" : Y, 
          "wt" : (len(self.model_parameters[self.name]["true_masses"])*(float(self.model_parameters[self.name]["toy_signal_events"])/self.array_size)*np.ones(self.array_size))
        },
        dtype=np.float64
      )

      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"true_mass"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(np.sum(df.loc[(df.loc[:,"true_mass"] == mass), "wt"]))

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}_Gaussian.parquet"
      pq.write_table(table, parquet_file_path)

      n_bkg_events = int(self.array_size/len(self.model_parameters[self.name]["true_masses"]))
      X_bkg = np.zeros(n_bkg_events)
      for ind in range(n_bkg_events):
        x = np.random.exponential(scale=1/self.model_parameters[self.name]["background_lambda"]) + self.model_parameters[self.name]["background_constant"]
        while x < self.model_parameters[self.name]["background_ranges"][0] or x > self.model_parameters[self.name]["background_ranges"][1]:
          x = np.random.exponential(scale=1/self.model_parameters[self.name]["background_lambda"]) + self.model_parameters[self.name]["background_constant"]
        X_bkg[ind] = x

      df = pd.DataFrame(
        {
          "reconstructed_mass" : X_bkg.flatten(), 
          "wt" : (float(self.model_parameters[self.name]["toy_background_events"])/float(n_bkg_events))*np.ones(n_bkg_events)
        },
        dtype=np.float64
      )
      df = df.sample(frac=1, random_state=42)
      df = df.reset_index(drop=True)

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}_ExpBkg.parquet"
      pq.write_table(table, parquet_file_path)

    elif self.name == "2D":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      X1 = np.random.normal(Y, self.model_parameters[self.name]["signal_resolution"]*Y, size = self.array_size)
      k_chi_1 = self.model_parameters[self.name]["chi_1"] + (Y - 173.0) * 0.1
      X2 = np.random.chisquare(df=k_chi_1, size=self.array_size)

      df = pd.DataFrame(
        {
          "X1": X1,
          "X2": X2, 
          "Y": Y,
          "wt": (len(self.model_parameters[self.name]["true_masses"])*(float(self.model_parameters[self.name]["toy_signal_events"])/self.array_size)*np.ones(self.array_size))
        },
        dtype=np.float64
      )
      # Rescale weights so all are equivalent
      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"Y"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(np.sum(df.loc[(df.loc[:,"Y"] == mass), "wt"]))

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}.parquet"
      pq.write_table(table, parquet_file_path)

    elif self.name == "5D":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      X1 = np.random.normal(Y, self.model_parameters[self.name]["signal_resolution"]*Y, size = self.array_size)
      X2 = np.random.exponential(Y/self.model_parameters[self.name]["exponential_factor_1"])
      X3 = np.random.exponential(Y/self.model_parameters[self.name]["exponential_factor_2"])
      X4 = np.random.exponential(Y/self.model_parameters[self.name]["exponential_factor_3"])

      # von mises cases
      mu = 0.0
      kappa_v = (1/ (self.model_parameters[self.name]["kappa"] * Y[0] * 0.1))**10 * (10**20)
      X5 = vonmises.rvs(kappa_v, loc=mu, size=self.array_size)

      df = pd.DataFrame(
        {
          "X1": X1,
          "X2": X2,
          "X3": X3, 
          "X4": X4,
          "X5": X5,
          "Y": Y,
          "wt": (len(self.model_parameters[self.name]["true_masses"])*(float(self.model_parameters[self.name]["toy_signal_events"])/self.array_size)*np.ones(self.array_size))
        },
        dtype=np.float64
      )

      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"Y"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(np.sum(df.loc[(df.loc[:,"Y"] == mass), "wt"]))

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}.parquet"
      pq.write_table(table, parquet_file_path)

    elif self.name == "5D+D":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)

      # gaussian
      X1 = np.random.normal(Y, self.model_parameters[self.name]["signal_resolution"]*Y, size = self.array_size)
     
      # chi
      k_chi = self.model_parameters[self.name]["chi"] + (Y - 173.0) * 0.1
      X2 = np.random.chisquare(df = k_chi, size=self.array_size)

      #exponential
      beta_e = self.model_parameters[self.name]["exponential_factor"] + Y * self.model_parameters[self.name]["exponential_factor"] / ((Y - 160.0)**2)
      X3 = np.random.exponential(beta_e) 

      #beta
      alpha_1 = self.model_parameters[self.name]["alpha"] + Y[0] * 0.02
      beta_1 = self.model_parameters[self.name]["beta"] * (Y[0] - 160.0) * 0.1
      X4 = beta.rvs(alpha_1, beta_1, size=self.array_size)

      #weibull
      lambda_w, k_w = Y * 0.05, (Y - 160.0) ** 2.0
      X5 = np.random.weibull(k_w, self.array_size) * lambda_w

      #discrete
      X6 = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], self.array_size, p=[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
      
      df = pd.DataFrame(
        {
          "X1": X1,
          "X2": X2,
          "X3": X3, 
          "X4": X4,
          "X5": X5,
          "X6": X6,
          "Y": Y,
          "wt": (len(self.model_parameters[self.name]["true_masses"])*(float(self.model_parameters[self.name]["toy_signal_events"])/self.array_size)*np.ones(self.array_size))
        },
        dtype=np.float64
      )

      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"Y"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(np.sum(df.loc[(df.loc[:,"Y"] == mass), "wt"]))
    
      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}.parquet"
      pq.write_table(table, parquet_file_path)

    elif self.name == "12D":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      X1 = np.random.normal(Y, self.model_parameters[self.name]["signal_resolution"]*Y, size = self.array_size)
      X2 = np.random.exponential(self.model_parameters[self.name]["exponential_factor_1"] + Y * self.model_parameters[self.name]["exponential_factor_1"] / ((Y - 160.0)**2))
      X3 = np.random.exponential(self.model_parameters[self.name]["exponential_factor_2"] + Y * self.model_parameters[self.name]["exponential_factor_2"] / ((Y - 160.0)**2))
      X4 = np.random.exponential(self.model_parameters[self.name]["exponential_factor_3"] + Y * self.model_parameters[self.name]["exponential_factor_3"] / ((Y - 160.0)**2))
      
      # von mises cases
      mu = 0.0 
      kappa_v = (1/ (self.model_parameters[self.name]["kappa"] * np.mean(Y) * 0.1))**10 * (10**15)
      X5 = vonmises.rvs(kappa_v, loc=mu, size=self.array_size)
      kappa_v2 = (1 / (self.model_parameters[self.name]["kappa_2"] * np.mean(Y) * 0.1))**10 * (10**10)
      X6 = vonmises.rvs(kappa_v2, loc=mu, size=self.array_size)
          
      # Beta distributions for X7 and X8
      alpha_7 = self.model_parameters[self.name]["alpha_7"] + Y * 0.02
      beta_7 = self.model_parameters[self.name]["beta_7"] * (Y - 160.0) * 0.1
      X7 = beta.rvs(alpha_7, beta_7, size=self.array_size)

      alpha_8 = self.model_parameters[self.name]["alpha_8"] + Y * 0.01
      beta_8 = self.model_parameters[self.name]["beta_8"] * (Y - 165.0) * 0.1
      X8 = beta.rvs(alpha_8, beta_8, size=self.array_size)

      # Chi-square distributions for X9 and X10
      k_chi_1 = self.model_parameters[self.name]["chi_1"] + (Y - 173.0) * 0.1
      X9 = np.random.chisquare(df=k_chi_1, size=self.array_size)

      k_chi_2 = self.model_parameters[self.name]["chi_2"] + (Y - 173.0) * 0.1
      X10 = np.random.chisquare(df=k_chi_2, size=self.array_size)

      # Weibull distribution for X11
      lambda_w, k_w = Y * 0.05 , (Y - 160.0) ** 2.0
      X11 = np.random.weibull(k_w, self.array_size) *  lambda_w

      # Another Gaussian distribution for X12
      std_dev_1 = self.model_parameters[self.name]["signal_resolution_1"] * Y
      X12 = np.random.normal(Y, std_dev_1, size=self.array_size)

      df = pd.DataFrame(
        {
          "X1": X1,
          "X2": X2,
          "X3": X3, 
          "X4": X4,
          "X5": X5,
          "X6": X6,
          "X7": X7,
          "X8": X8,
          "X9": X9,
          "X10": X10,
          "X11": X11,
          "X12": X12,
          "Y": Y,
          "wt": (len(self.model_parameters[self.name]["true_masses"])*(float(self.model_parameters[self.name]["toy_signal_events"])/self.array_size)*np.ones(self.array_size))
        },
        dtype=np.float64
      )

      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"Y"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(np.sum(df.loc[(df.loc[:,"Y"] == mass), "wt"]))
      
      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}.parquet"
      pq.write_table(table, parquet_file_path)

    elif ".yaml" in self.name:

      events_per_file = 10**5

      for file_name in self.benchmark_info["cfg"]["files"].keys():

        unique_vals = list(np.unique(self.benchmark_info["cfg"]["preprocess"]["train_test_y_vals"][self.benchmark_info["cfg"]["pois"][0]] + self.benchmark_info["cfg"]["preprocess"]["validation_y_vals"][self.benchmark_info["cfg"]["pois"][0]]))
        if list(self.benchmark_info["inverse_cdf_splines"][file_name].keys())[0] == "all":
          unique_vals = ["all"]

        for ind, key in enumerate(unique_vals):

          data = self.SampleFromSplines(key, file_name, n_samples=events_per_file)

          if key in self.benchmark_info["yield"][file_name].keys():
            total_yield = self.benchmark_info["yield"][file_name][key]
          else:
            uvs = list(self.benchmark_info["yield"][file_name].keys())
            uvs_p = sorted(uvs + [key])
            uvs_ind = uvs_p.index(key)
            lower_y = uvs[uvs_ind-1]
            higher_y = uvs[uvs_ind]
            lower_data = self.benchmark_info["yield"][file_name][lower_y]
            higher_data = self.benchmark_info["yield"][file_name][higher_y]
            frac = (key-lower_y)/(higher_y-lower_y)
            total_yield = lower_data + frac*(higher_data-lower_data)            

          wts = (total_yield/events_per_file) * np.ones((events_per_file,1))

          if not key == "all": 
            columns = [
              self.benchmark_info["cfg"]["variables"][0],
              self.benchmark_info["cfg"]["pois"][0],
              "wt"
            ]
            data = np.concatenate((data.reshape(-1,1),key*np.ones((events_per_file,1)),wts), axis=1)
          else:
            columns = [
              self.benchmark_info["cfg"]["variables"][0],
              "wt"
            ]
            data = np.concatenate((data.reshape(-1,1),wts), axis=1)

          if ind == 0:
            df = pd.DataFrame(data, columns=columns, dtype=np.float64)
          else:
            temp_df = pd.DataFrame(data, columns=columns, dtype=np.float64)
            df = pd.concat([df,temp_df], ignore_index=True)

        table = pa.Table.from_pandas(df)
        parquet_file_path = f"data/{self.name.split('/')[-1].split('.yaml')[0]}_{file_name}.parquet"
        pq.write_table(table, parquet_file_path)

  def MakeConfig(self, return_cfg=False):
    """
    Generate configuration files for benchmarks.
    """
    if self.name == "Gaussian" or self.name == "GaussianWithExpBkg":

      cfg = {
        "name" : f"Benchmark_{self.name}",
        "files" : {self.name : f"data/{self.name}.parquet"},
        "variables" : ["reconstructed_mass"],
        "pois" : ["true_mass"],
        "nuisances" : [],
        "preprocess" : {
          "standardise" : "all",
          "train_test_val_split" : "0.3:0.3:0.4",
          "equalise_y_wts" : True,
          "train_test_y_vals" : {
            "true_mass" : [
              166.0,167.0,168.0,169.0,170.0,
              171.0,172.0,173.0,174.0,175.0,
              176.0,177.0,178.0,179.0,180.0,              
            ]
          },
          "validation_y_vals" : {
            "true_mass" : [
              171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
            ]
          }
        },
        "inference" : {},
        "validation" : {},
        "data_file" : None
      }

    elif self.name == "GaussianWithExpBkgVaryingYield":

      cfg = {
        "name" : f"Benchmark_{self.name}",
        "files" : {
          "Gaussian" : f"data/{self.name}_Gaussian.parquet",
          "ExpBkg" : f"data/{self.name}_ExpBkg.parquet"
          },
        "variables" : ["reconstructed_mass"],
        "pois" : ["true_mass"],
        "nuisances" : [],
        "preprocess" : {
          "standardise" : "all",
          "train_test_val_split" : "0.3:0.3:0.4",
          "equalise_y_wts" : True,
          "train_test_y_vals" : {
            "true_mass" : [
              166.0,167.0,168.0,169.0,170.0,
              171.0,172.0,173.0,174.0,175.0,
              176.0,177.0,178.0,179.0,180.0,              
            ]
          },
          "validation_y_vals" : {
            "true_mass" : [
              171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
            ]
          }
        },
        "inference" : {
          "rate_parameters" : ["Gaussian"]
        },
        "validation" : {
          "rate_parameter_vals" : {
            "Gaussian" : [0.2,0.4,0.6]
          }
        },
        "data_file" : None
      }      

    elif self.name == "2D":

      cfg = {
        "name" : f"Benchmark_{self.name}",
        "files" : {self.name : f"data/{self.name}.parquet"},
        "variables" : ["X1", "X2"],
        "pois" : ["Y"],
        "nuisances" : [],
        "preprocess" : {
          "standardise" : "all",
          "train_test_val_split" : "0.3:0.3:0.4",
          "equalise_y_wts" : True,
          "train_test_y_vals" : {
            "Y" : [
              166.0,167.0,168.0,169.0,170.0,
              171.0,172.0,173.0,174.0,175.0,
              176.0,177.0,178.0,179.0,180.0,              
            ]
          },
          "validation_y_vals" : {
            "Y" : [
              171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
            ]
          }
        },
        "inference" : {},
        "validation" : {},
        "data_file" : None
      }

    elif self.name == "5D":
        
      cfg = {
      "name" : f"Benchmark_{self.name}",
      "files" : {self.name : f"data/{self.name}.parquet"},
      "variables" : ["X1", "X2", "X3", "X4", "X5"],
      "pois" : ["Y"],
      "nuisances" : [],
      "preprocess" : {
        "standardise" : "all",
        "train_test_val_split" : "0.3:0.3:0.4",
        "equalise_y_wts" : True,
        "train_test_y_vals" : {
          "Y" : [
            166.0,167.0,168.0,169.0,170.0,
            171.0,172.0,173.0,174.0,175.0,
            176.0,177.0,178.0,179.0,180.0,              
          ]
        },
        "validation_y_vals" : {
          "Y" : [
            171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          ]
        }
      },
      "inference" : {},
      "validation" : {},
      "data_file" : None
    }

    elif self.name == "5D+D":

      cfg = {
      "name" : f"Benchmark_{self.name}",
      "files" : {self.name : f"data/{self.name}.parquet"},
      "variables" : ["X1", "X2", "X3", "X4", "X5", "X6"],
      "pois" : ["Y"],
      "nuisances" : [],
      "preprocess" : {
        "standardise" : "all",
        "train_test_val_split" : "0.3:0.3:0.4",
        "equalise_y_wts" : True,
        "train_test_y_vals" : {
          "Y" : [
            166.0,167.0,168.0,169.0,170.0,
            171.0,172.0,173.0,174.0,175.0,
            176.0,177.0,178.0,179.0,180.0,              
          ]
        },
        "validation_y_vals" : {
          "Y" : [
            171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          ]
        }
      },
      "inference" : {},
      "validation" : {},
      "data_file" : None
    }

    elif self.name == "12D":
        
      cfg = {
      "name" : f"Benchmark_{self.name}",
      "files" : {self.name : f"data/{self.name}.parquet"},
      "variables" : ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12"],
      "pois" : ["Y"],
      "nuisances" : [],
      "preprocess" : {
        "standardise" : "all",
        "train_test_val_split" : "0.3:0.3:0.4",
        "equalise_y_wts" : True,
        "train_test_y_vals" : {
          "Y" : [
            166.0,167.0,168.0,169.0,170.0,
            171.0,172.0,173.0,174.0,175.0,
            176.0,177.0,178.0,179.0,180.0,              
          ]
        },
        "validation_y_vals" : {
          "Y" : [
            171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          ]
        }
      },
      "inference" : {},
      "validation" : {},
      "data_file" : None
    }


    if return_cfg:
      return cfg

    if not ".yaml" in self.name:

      with open(f"configs/run/Benchmark_{self.name}.yaml", 'w') as file:
        yaml.dump(cfg, file)

    else:

      cfg = copy.deepcopy(self.benchmark_info["cfg"])
      name = self.name.split('/')[-1].split('.yaml')[0]
      cfg["name"] = "Benchmark_"+name

      for f in self.benchmark_info["cfg"]["files"].keys():
        cfg["files"][f] = f"data/{name}_{f}.parquet"

      cfg["preprocess"]["selection"] = None

      with open(f"configs/run/Benchmark_{name}.yaml", 'w') as file:
        yaml.dump(cfg, file)    



  def FitSplines(self, n_bins=10000):

    from data_loader import DataLoader
    from scipy.interpolate import CubicSpline

    # Define storage for information
    self.benchmark_info = {}
    self.benchmark_info["yield"] = {}
    self.benchmark_info["pdf_splines"] = {}
    self.benchmark_info["cdf_splines"] = {}
    self.benchmark_info["inverse_cdf_splines"] = {}

    # Open config
    with open(self.name, 'r') as yaml_file:
      self.benchmark_info["cfg"] = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loop through files
    for file_name, f in self.benchmark_info["cfg"]["files"].items():
      dl = DataLoader(f)
      dl.ChangeBatchSize(100000)
      data = dl.LoadFullDataset()

      # Applying selection
      if self.benchmark_info["cfg"]["preprocess"]["selection"] is not None:
        data = data.loc[data.eval(self.benchmark_info["cfg"]["preprocess"]["selection"]),:]

      # Get datasets to get histgrams for
      if self.benchmark_info["cfg"]["pois"][0] not in data.keys():
        uvs = []
        datasets = [data.loc[:,[self.benchmark_info["cfg"]["variables"][0], "wt"]].to_numpy()]
      else:
        uvs = sorted([float(i) for i in data.loc[:,self.benchmark_info["cfg"]["pois"][0]].drop_duplicates().to_numpy()])
        datasets = []
        for uv in uvs:
          datasets.append(data.loc[(data.loc[:,self.benchmark_info["cfg"]["pois"][0]] == uv), [self.benchmark_info["cfg"]["variables"][0], "wt"]].to_numpy())

      # Make hists and get yields
      self.benchmark_info["yield"][file_name] = {}
      hists = []
      bins = []
      for ind, ds in enumerate(datasets):

        hist, b = np.histogram(ds[:,0], bins=n_bins, weights=ds[:,1], density=True)
        hists.append(hist)
        bins.append(b)

        if len(uvs) == 0:
          key = "all"
        else:
          key = uvs[ind]
        
        self.benchmark_info["yield"][file_name][key] = float(np.sum(ds[:,1]))

        bin_midpoints = (b[:-1] + b[1:]) / 2
        mean = np.average(bin_midpoints, weights=hist)

      del data, datasets

      # Fit splines
      self.benchmark_info["pdf_splines"][file_name] = {}
      self.benchmark_info["cdf_splines"][file_name] = {}
      self.benchmark_info["inverse_cdf_splines"][file_name] = {}

      for ind, h in enumerate(hists):
        if len(uvs) == 0:
          key = "all"
        else:
          key = uvs[ind]

        bin_midpoints = (bins[ind][:-1] + bins[ind][1:]) / 2
        cdf_hist = np.cumsum(h)
        cdf_hist = cdf_hist / cdf_hist[-1]

        self.benchmark_info["pdf_splines"][file_name][key] = CubicSpline(bin_midpoints, h)
        unique_bins, unique_indices = np.unique(bin_midpoints, return_index=True)
        unique_cdf = cdf_hist[unique_indices]
        self.benchmark_info["cdf_splines"][file_name][key] = CubicSpline(unique_bins, unique_cdf)
        unique_cdf, unique_indices = np.unique(cdf_hist, return_index=True)
        unique_bins = bin_midpoints[unique_indices]       
        self.benchmark_info["inverse_cdf_splines"][file_name][key] = CubicSpline(unique_cdf, unique_bins)  

  def SampleFromSplines(self, y, file_name, n_samples=100):

    uniform_values = np.random.uniform(low=0, high=1, size=n_samples)
    if y == "all":
      data = self.benchmark_info["inverse_cdf_splines"][file_name]["all"](uniform_values)
    else:
      # check if in file
      if y in self.benchmark_info["inverse_cdf_splines"][file_name].keys():
        data = self.benchmark_info["inverse_cdf_splines"][file_name][y](uniform_values)
      else:
        # find nearest neighbours
        uvs = list(self.benchmark_info["inverse_cdf_splines"][file_name].keys())
        uvs_p = sorted(uvs + [y])
        ind = uvs_p.index(y)
        lower_y = uvs[ind-1]
        higher_y = uvs[ind]
        lower_data = self.benchmark_info["inverse_cdf_splines"][file_name][lower_y](uniform_values)
        higher_data = self.benchmark_info["inverse_cdf_splines"][file_name][higher_y](uniform_values)
        frac = (y-lower_y)/(higher_y-lower_y)
        data = lower_data + frac*(higher_data-lower_data)
    return data




    