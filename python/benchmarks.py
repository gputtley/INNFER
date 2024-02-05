import yaml
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
      "5D+D": {
        "signal_resolution" : 0.01, # guassian
        "chi" : 5.0, # chi-squared
        "alpha" : 0.5, # beta 
        "beta" : 2.0, # beta
        "exponential_factor" : 20, # exponential  # Weibull
        "true_masses" : [
          166.0,167.0,168.0,169.0,170.0,
          170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,
          176.0,177.0,178.0,179.0,180.0,
        ],    
        "toy_signal_events" : 1000.0,
      },
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

    elif file_name == "5D+D":
      #gaussian
      std = self.model_parameters[self.name]["signal_resolution"] * Y[0]
      mean = Y[0]
      gaussian_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-X[0] - mean) ** 2 / (2 * std**2)

      #chi
      k_chi = self.model_parameters[self.name]["chi"] + (Y[0] - 173.0) * 0.1
      chi_pdf = ((1/2)**(k_chi/2)) / gamma(k_chi/2) * (X[1]**(k_chi/2 - 1)) * np.exp(-X[1]/2)

      #exponential
      beta_e = self.model_parameters[self.name]["exponential_factor"] + Y[0] * self.model_parameters[self.name]["exponential_factor_1"] / ((Y[0] - 160.0)**2)
      exponential_pdf = (1/beta_e)*np.exp(-X[2]/beta_e)

      #beta
      alpha_1 = self.model_parameters[self.name]["alpha"] + Y[0] * 0.02
      beta_1 = self.model_parameters[self.name]["beta"] * (Y[0] - 160.0) * 0.1
      beta_pdf = beta.pdf(X[3], alpha_1, beta_1)

      #weibull
      lambda_w, k_w = Y[0] * 0.05, (Y[0] - 160.0) ** 2.0
      weibull_pdf = (k_w / lambda_w) * (X[4] / lambda_w)**(k_w - 1) * np.exp(-(X[4] / lambda_w)**k_w)

      #discrete
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
        }
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
        }
      )
      # Rescale weights so all are equivalent
      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"true_mass"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(self.model_parameters[self.name]["signal_fraction"] * np.sum(df.loc[(df.loc[:,"true_mass"] == mass), "wt"]))

      df = df.sample(frac=1, random_state=42)
      df = df.reset_index(drop=True)

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}.parquet"
      pq.write_table(table, parquet_file_path)

    elif self.name == "GaussianWithExpBkgVaryingYield":

      print(">> Making signal events")
      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      X = np.random.normal(Y, self.model_parameters[self.name]["signal_resolution"]*Y)
      df = pd.DataFrame(
        {
          "reconstructed_mass" : X, 
          "true_mass" : Y, 
          "wt" : (len(self.model_parameters[self.name]["true_masses"])*(float(self.model_parameters[self.name]["toy_signal_events"])/self.array_size)*np.ones(self.array_size))
        }
      )

      # Rescale weights so all are equivalent
      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"true_mass"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(np.sum(df.loc[(df.loc[:,"true_mass"] == mass), "wt"]))

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}_Gaussian.parquet"
      pq.write_table(table, parquet_file_path)

      print(">> Making background events")
      report_percentile = 0.05
      n_bkg_events = int(self.array_size/len(self.model_parameters[self.name]["true_masses"]))
      X_bkg = np.zeros(n_bkg_events)
      for ind in range(n_bkg_events):
        x = np.random.exponential(scale=1/self.model_parameters[self.name]["background_lambda"]) + self.model_parameters[self.name]["background_constant"]
        while x < self.model_parameters[self.name]["background_ranges"][0] or x > self.model_parameters[self.name]["background_ranges"][1]:
          x = np.random.exponential(scale=1/self.model_parameters[self.name]["background_lambda"]) + self.model_parameters[self.name]["background_constant"]
        X_bkg[ind] = x
        if ind % np.ceil(n_bkg_events*report_percentile) == 0:
          print(f"{100*round(float(ind)/n_bkg_events,2)}% Finished")
      df = pd.DataFrame(
        {
          "reconstructed_mass" : X_bkg.flatten(), 
          "wt" : (float(self.model_parameters[self.name]["toy_background_events"])/float(n_bkg_events))*np.ones(n_bkg_events)
        }
      )
      df = df.sample(frac=1, random_state=42)
      df = df.reset_index(drop=True)

      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}_ExpBkg.parquet"
      pq.write_table(table, parquet_file_path)

    elif self.name == "5D+D":
      print(">> Making signal events with the 5D plus one discrete X-space")
      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      # guassain
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
      
      # generate dataframe
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
        }
      )

      # Rescale weights so all are equivalent
      for mass in self.model_parameters[self.name]["true_masses"]:
        df.loc[(df.loc[:,"Y"] == mass), "wt"] *= float(self.model_parameters[self.name]["toy_signal_events"]) / float(np.sum(df.loc[(df.loc[:,"Y"] == mass), "wt"]))
    
    
      table = pa.Table.from_pandas(df)
      parquet_file_path = f"data/{self.name}.parquet"
      pq.write_table(table, parquet_file_path)

  def MakeConfig(self):
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

    with open(f"configs/run/Benchmark_{self.name}.yaml", 'w') as file:
      yaml.dump(cfg, file)


