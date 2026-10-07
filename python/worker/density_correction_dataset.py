import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_loader import DataLoader
from useful_functions import MakeDirectories


class DensityCorrectionDataset():
  """
  Build matched weighted simulation/flow classification tables in training space.
  """

  def __init__(self):

    self.data_input = "data/"
    self.synthetic_input = "data/"
    self.data_output = "data/"
    self.data_parameters = {}
    self.batch_size = 64
    self.seed = 42


  def Configure(self, options):
    """
    Configure the dataset settings.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if not isinstance(self.batch_size, int) or self.batch_size < 1:
      raise ValueError("batch_size must be a positive integer")
    if self.seed < 0:
      raise ValueError("seed must be non-negative")


  def Build(self, source_split, output_split, split_index=0):
    """
    Copy each source condition and signed weight into both classes. Class 1 is
    simulation; class 0 uses the matching EvaluateDensity row with the same weight.
    """
    X_columns = self.data_parameters["X_columns"]
    Y_columns = self.data_parameters["Y_columns"]
    feature_columns = X_columns + Y_columns
    if len(set(feature_columns)) != len(feature_columns):
      raise ValueError("Observable and condition columns must be distinct")
    if any(column in feature_columns for column in ["classifier_truth", "wt"]):
      raise ValueError("classifier_truth and wt are reserved dataset columns")

    loaders = {
      key: DataLoader(f"{self.data_input}/{key}_{source_split}.parquet", batch_size=self.batch_size)
      for key in ["X", "wt"] + (["Y"] if Y_columns else [])
    }
    synthetic_file = f"{self.synthetic_input}/synth_{output_split}.parquet"
    loaders["synthetic"] = DataLoader(synthetic_file, batch_size=self.batch_size)
    n_rows = loaders["X"].num_rows
    if n_rows == 0 or any(loader.num_rows != n_rows for loader in loaders.values()):
      raise ValueError(f"Empty or misaligned density source/EvaluateDensity tables for {source_split}")
    rng = np.random.default_rng(self.seed + split_index)
    writers = {}
    sums = {"simulation": 0.0, "synthetic": 0.0}
    negative_rows = 0

    try:
      for start in range(0, n_rows, self.batch_size):
        n_batch = min(self.batch_size, n_rows-start)
        X = loaders["X"].LoadNextBatch().iloc[:n_batch].reset_index(drop=True).loc[:, X_columns]
        Y = loaders["Y"].LoadNextBatch().iloc[:n_batch].reset_index(drop=True).loc[:, Y_columns] if Y_columns else pd.DataFrame(index=range(n_batch))
        weights = loaders["wt"].LoadNextBatch().iloc[:n_batch].reset_index(drop=True)["wt"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(X.to_numpy())) or not np.all(np.isfinite(Y.to_numpy())) or not np.all(np.isfinite(weights)):
          raise ValueError(f"Non-finite density training data in {source_split}")

        simulation = pd.concat([X, Y], axis=1)
        simulation["classifier_truth"] = np.ones(n_batch, dtype=np.int64)
        simulation["wt"] = weights
        generated = loaders["synthetic"].LoadNextBatch().reset_index(drop=True)
        if Y_columns and set(Y_columns).issubset(generated.columns):
          if not np.array_equal(generated.loc[:, Y_columns].to_numpy(), Y.to_numpy()):
            raise ValueError(f"EvaluateDensity conditions do not match the source rows for {source_split}")
        synthetic = generated.loc[:, X_columns]
        if len(synthetic) != n_batch or not np.all(np.isfinite(synthetic.to_numpy())):
          raise ValueError("EvaluateDensity must provide one finite row per source condition")
        synthetic = pd.concat([synthetic, Y.copy(deep=True)], axis=1)
        synthetic["classifier_truth"] = np.zeros(n_batch, dtype=np.int64)
        synthetic["wt"] = weights

        batch = pd.concat([simulation, synthetic], ignore_index=True)
        batch = batch.iloc[rng.permutation(len(batch))].reset_index(drop=True)
        tables = {
          f"X_{output_split}": batch.loc[:, feature_columns].astype(np.float64),
          f"y_{output_split}": batch.loc[:, ["classifier_truth"]],
          f"wt_{output_split}": batch.loc[:, ["wt"]],
        }
        for name, frame in tables.items():
          table = pa.Table.from_pandas(frame, preserve_index=False)
          if name not in writers:
            path = f"{self.data_output}/{name}.parquet"
            MakeDirectories(path)
            writers[name] = pq.ParquetWriter(path, table.schema, compression="snappy")
          writers[name].write_table(table)

        sums["simulation"] += float(weights.sum())
        sums["synthetic"] += float(weights.sum())
        negative_rows += int(np.count_nonzero(weights < 0))
    finally:
      for writer in writers.values():
        writer.close()
      for loader in loaders.values():
        if loader.parquet_file is not None:
          loader.parquet_file.close()

    if sums["simulation"] <= 0:
      raise ValueError(f"Density correction requires a positive signed weight sum for {source_split}")
    return {
      "source_split": source_split,
      "synthetic_file": synthetic_file,
      "source_rows": n_rows,
      "simulation_rows": n_rows,
      "synthetic_rows": n_rows,
      "negative_simulation_rows": negative_rows,
      "sum_weights": sums,
    }
