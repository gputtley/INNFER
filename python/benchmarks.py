import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class Benchmarks():

  def __init__(self, name="Gaussian", parameters={}):

    self.name = name
    self.model_parameters = {
      "Gaussian": {
        "signal_resolution" : 0.1,
        "true_masses" : [170.0,171.0,172.0,173.0,174.0,175.0],
      },
      "GaussianWithExpBkg": {
        "signal_resolution" : 0.1,
        "signal_fraction" : 0.3,
        "background_ranges" : [160.0,185.0],
        "background_lambda" : 0.1,
        "background_constant" : 160.0,
        "true_masses" : [170.0,171.0,172.0,173.0,174.0,175.0],
      }
    }
    self.saved_parameters = {}
    self.array_size = 10**6

    for k, v in parameters.items():
      self.model_parameters[name][k] = v


  def GetPDF(self, X, Y):

    if isinstance(X, float) or isinstance(X, int): X = np.array([X])
    if isinstance(Y, float) or isinstance(Y, int): Y = np.array([Y])

    if self.name == "Gaussian":

      std_dev = self.model_parameters[self.name]["signal_resolution"] * Y[0]
      mean = Y[0]
      return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(X[0] - mean)**2 / (2 * std_dev**2))
    
    elif self.name == "GaussianWithExpBkg":

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

  def MakeDataset(self):

    if self.name == "Gaussian":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      X = np.random.normal(Y, self.model_parameters[self.name]["signal_resolution"]*Y)
      df = pd.DataFrame({"reconstructed_mass" : X, "true_mass" : Y, "wt" : np.ones(len(X))})

    elif self.name == "GaussianWithExpBkg":

      Y = np.random.choice(self.model_parameters[self.name]["true_masses"], size=self.array_size)
      signal_entries = int(round(self.array_size*self.model_parameters[self.name]["signal_fraction"]))
      X_signal = np.random.normal(Y[:signal_entries], self.model_parameters[self.name]["signal_resolution"]*Y[:signal_entries])
      bkg_entries = self.array_size - signal_entries
      X_bkg = np.array([])
      for _ in range(bkg_entries):
        x = np.random.exponential(scale=1/self.model_parameters[self.name]["background_lambda"]) + self.model_parameters[self.name]["background_constant"]
        while x < self.model_parameters[self.name]["background_ranges"][0] or x > self.model_parameters[self.name]["background_ranges"][1]:
          x = np.random.exponential(scale=1/self.model_parameters[self.name]["background_lambda"]) + self.model_parameters[self.name]["background_constant"]
        X_bkg = np.append(X_bkg, x)
      X = np.vstack((X_signal.reshape(-1,1),X_bkg.reshape(-1,1)))
      df = pd.DataFrame({"reconstructed_mass" : X.flatten(), "true_mass" : Y.flatten(), "wt" : np.ones(len(X))})
      df = df.sample(frac=1, random_state=42)
      df = df.reset_index(drop=True)

    table = pa.Table.from_pandas(df)
    parquet_file_path = f"data/{self.name}.parquet"
    pq.write_table(table, parquet_file_path)

  def MakeConfig(self):

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
          "equalise_y_wts" : False,          
        },
        "inference" : {},
        "data_file" : None
      }

    with open(f"configs/run/Benchmark_{self.name}.yaml", 'w') as file:
      yaml.dump(cfg, file)


