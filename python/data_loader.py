import numpy as np
import pandas as pd
import pyarrow.parquet as pq

class DataLoader():

  def __init__(self, parquet_file_name, batch_size=1024):
    """
    Initialize the DataLoader.

    Args:
        parquet_file_name (str): The name of the Parquet file to load.
        batch_size (int, optional): The batch size. Default is 1024.
    """
    self.parquet_file_name = parquet_file_name
    self.parquet_file = pq.ParquetFile(parquet_file_name)
    self.batch_size = batch_size
    self.generator = None
    self.batch_num = 0
    self.num_rows = self.parquet_file.metadata.num_rows
    self.num_columns = len(self.parquet_file.schema.names)
    self.num_batches = int(np.ceil(self.num_rows/self.batch_size))

  def LoadNextBatch(self):
    """
    Load the next batch of data.

    Returns:
        pd.DataFrame: The next batch of data as a Pandas DataFrame.
    """
    if self.batch_num == 0:
      self.generator = self.parquet_file.iter_batches(batch_size=self.batch_size)
    self.batch_num = (self.batch_num + 1) % self.num_batches
    return next(self.generator).to_pandas()

  def LoadFullDataset(self):
    """
    Load the full dataset by iterating through batches.

    Returns:
        pd.DataFrame: The full dataset as a Pandas DataFrame.
    """
    data = self.LoadNextBatch()
    for _ in range(1, self.num_batches):
      data = pd.concat([data,self.LoadNextBatch()], ignore_index=True)
    return data


    


