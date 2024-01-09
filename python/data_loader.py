import numpy as np
import pandas as pd
import pyarrow.parquet as pq

class DataLoader():
  """
  DataLoader class for loading datasets in from a parquet file.
  """
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
    self.columns = [str(i) for i in self.parquet_file.schema.names if i != '__index_level_0__']
    self.num_columns = len(self.columns)
    self.num_batches = int(np.ceil(self.num_rows/self.batch_size))

  def LoadNextBatch(self):
    """
    Load the next batch of data.

    Returns:
        pd.DataFrame: The next batch of data as a Pandas DataFrame.
    """
    if self.num_rows > 0:
      if self.batch_num == 0:
        self.generator = self.parquet_file.iter_batches(batch_size=self.batch_size)
      self.batch_num = (self.batch_num + 1) % self.num_batches
      return next(self.generator).to_pandas()
    else:
      return pd.DataFrame(index=range(self.batch_size))

  def LoadFullDataset(self):
    """
    Load the full dataset by iterating through batches.

    Returns:
        pd.DataFrame: The full dataset as a Pandas DataFrame.
    """
    orig_batch_size = int(self.batch_size)
    orig_num_batches = int(self.num_batches)
    self.batch_size = int(self.num_rows)
    self.num_batches = 1
    data = self.LoadNextBatch()
    for _ in range(1, self.num_batches):
      data = pd.concat([data,self.LoadNextBatch()], ignore_index=True)
    self.batch_size = int(orig_batch_size)
    self.num_batches = int(orig_num_batches)
    return data

  def ChangeBatchSize(self, batch_size):
    """
    Changes the size of the loader batch.

    Args:
        batch_size (int): The new batch size.
    """
    self.batch_size = batch_size
    self.num_batches = int(np.ceil(self.num_rows/self.batch_size))


