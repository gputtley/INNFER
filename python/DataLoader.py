import gc
import copy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from collections import OrderedDict

class DataLoader():

  def __init__(
      self, 
      parquet_file_name,
      batch_size = 1024
      ):

    # File information
    self.parquet_file_name = parquet_file_name
    self.parquet_file = pq.ParquetFile(parquet_file_name)
    self.batch_size = batch_size

    # Overwritten variables
    self.generator = None
    self.batch_num = 0

    # Dataset information
    self.num_rows = self.parquet_file.metadata.num_rows
    print(self.parquet_file.schema.names, len(self.parquet_file.schema.names))
    self.num_columns = len(self.parquet_file.schema.names)
    self.num_batches = int(np.ceil(self.num_rows/self.batch_size))

    # For pre-processing
    self.shuffle = True
    self.train_test_split = None
    self.feature_normalisation = False
    self.separate_variables = None # Example: {"context":["top_mass"], "x":["px_1","py_1","pz_1","px_2","py_2","pz_2"]}

    # For post-processing
    self.feature_normalisation_mean = None
    self.feature_normalisation_std = None

    # Other
    self.random_seed = 42

  def PreProcess(self):

    datasets = {"preprocessed": self.LoadFullDataset()}

    if self.shuffle:
      for key, data in copy.deepcopy(datasets).items():
        datasets["{}_shuffled".format(key)] = data.sample(frac=1.0, random_state=self.random_seed)
        datasets["{}_shuffled".format(key)].reset_index(drop=True, inplace=True)
        del datasets[key]
        gc.collect()  

    if self.train_test_split != None:
      for key, data in copy.deepcopy(datasets).items():
        train, test =  train_test_split(data, test_size=self.train_test_split, random_state=self.random_seed)
        datasets["{}_train".format(key)] = train
        datasets["{}_test".format(key)] = test
        del train, test, datasets[key]
        gc.collect()

    if self.separate_variables != None:
      for key, data in copy.deepcopy(datasets).items():
        for k, v in self.separate_variables.items():
          datasets["{}_{}".format(key,k)] = data.loc[:,v]
        del datasets[key]
        gc.collect()   

    if self.feature_normalisation:
      feature_mean = {}
      feature_std = {}
      for key, data in copy.deepcopy(datasets).items():
        if "_test" in key: continue
        name = "{}_normalised".format(key)
        feature_mean[name] = data.mean()
        feature_std[name] = data.std()
        datasets[name] = (data-feature_mean[name])/feature_std[name]
        del datasets[key]
        gc.collect()          
        if "_train" in key:
          test_name = "{}_normalised".format(key.replace("_train","_test"))
          feature_mean[test_name] = feature_mean[name]
          feature_std[test_name] = feature_std[name]
          datasets[test_name] = (data-feature_mean[test_name])/feature_std[test_name]
          del datasets[key.replace("_train","_test")]
          gc.collect()  

    return_dl = {}
    for k, v in datasets.items():
      name = self.parquet_file_name.replace(".parquet","_{}.parquet".format(k))
      self.parquet_file = v.to_parquet(name, index=False)
      return_dl[k] = DataLoader(name, batch_size=self.batch_size)
      if self.feature_normalisation:
        return_dl[k].feature_normalisation_mean = feature_mean[k]
        return_dl[k].feature_normalisation_std = feature_std[k]

    return return_dl

  def LoadNextBatch(self):

    if self.batch_num == 0:
      self.generator = self.parquet_file.iter_batches(batch_size=self.batch_size)

    if self.batch_num + 1 < self.num_batches:
      self.batch_num += 1
    else:
      self.batch_num = 0

    return next(self.generator).to_pandas()

  def LoadFullDataset(self):

    for batch_num in range(self.num_batches):
      if batch_num == 0:
        data = self.LoadNextBatch()
      else:
        data = pd.concat([data,self.LoadNextBatch()])

    return data


    


