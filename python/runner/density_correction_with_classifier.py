import copy
import os

import numpy as np
import yaml

from data_loader import DataLoader
from density_correction_dataset import DensityCorrectionDataset
from useful_functions import InitiateClassifierModel, MakeDirectories


class DensityCorrectionWithClassifier():
  """
  Train a conditional simulation-to-flow density ratio without applying it.
  """

  def __init__(self):

    self.parameters = None
    self.architecture = None
    self.file_name = None
    self.data_input = "data/"
    self.synthetic_input = "data/"
    self.density_model = None
    self.data_output = "data/"
    self.model_output = "models/"
    self.plots_output = "plots/"
    self.train_name = "train"
    self.test_name = "test"
    self.seed = 42
    self.disable_tqdm = False
    self.use_wandb = False
    self.wandb_project_name = "innfer"
    self.wandb_submit_name = "density_correction"
    self.save_model_per_epoch = False
    self.verbose = True


  def Configure(self, options):
    """
    Configure the class settings.
    """
    for key, value in options.items():
      setattr(self, key, value)


  def Run(self):

    with open(self.parameters, 'r') as file:
      parameters = yaml.safe_load(file)
    with open(self.architecture, 'r') as file:
      architecture = yaml.safe_load(file)
    with open(f"{self.density_model}_architecture.yaml", 'r') as file:
      density_architecture = yaml.safe_load(file)
    if architecture["type"] != "FCNN":
      raise ValueError("Density correction requires an FCNN classifier without nuisance interpolation")

    density_parameters = parameters["density"]
    feature_columns = density_parameters["X_columns"] + density_parameters["Y_columns"]
    selected_columns = architecture.get("only_X_columns")
    if selected_columns is not None and (not set(density_parameters["Y_columns"]).issubset(selected_columns) or not set(selected_columns).issubset(feature_columns)):
      raise ValueError("Classifier feature selection must include every density condition")
    # Paired class weights already have equal priors and identical condition
    # marginals. Batch-dependent class normalisation would change that target.
    architecture["normalise_batch_categories"] = False
    architecture["num_classes"] = 2
    architecture["task"] = "classification"
    batch_size = architecture.get("batch_size", 64)
    if not isinstance(batch_size, int) or batch_size < 1:
      raise ValueError("Classifier architecture batch_size must be a positive integer")
    architecture["batch_size"] = batch_size

    builder = DensityCorrectionDataset()
    builder.Configure({
      "data_input": self.data_input, "data_output": self.data_output,
      "synthetic_input": self.synthetic_input,
      "data_parameters": density_parameters,
      "seed": self.seed, "batch_size": batch_size,
    })
    report = {}
    for split_index, (source_split, output_split) in enumerate([(self.train_name, "train"), (self.test_name, "test")]):
      if self.verbose:
        print(f"- Building matched {output_split} classification data")
      report[output_split] = builder.Build(source_split, output_split, split_index=split_index)

    classifier_parameters = {
      "X_columns": feature_columns, "y_columns": ["classifier_truth"],
      "standardisation": {}, "feature_space": "density_transformed",
      "file_loc": self.data_output,
    }
    metadata = {
      "file_name": self.file_name, "density_model": self.density_model,
      "synthetic_input": self.synthetic_input,
      "density_parameters": copy.deepcopy(density_parameters),
      "density_architecture": copy.deepcopy(density_architecture),
      "classifier": classifier_parameters,
      "classes": {0: "synthetic", 1: "simulation"},
      "likelihood_ratio": "P(class=1|X,Y) / P(class=0|X,Y)",
      "ratio_target": "p_simulation(X|Y) / p_flow(X|Y)",
      "equal_class_weights": True, "signed_weights": True,
      "seed": self.seed, "batch_size": batch_size,
    }
    self._WriteYaml(f"{self.data_output}/parameters.yaml", metadata)
    self._WriteYaml(f"{self.model_output}/{self.file_name}_architecture.yaml", architecture)

    if self.use_wandb:
      import wandb
      wandb.init(project=self.wandb_project_name, name=self.wandb_submit_name, config=architecture)
    if self.verbose:
      print("- Training the density correction classifier")
    classifier = InitiateClassifierModel(
      architecture, self.data_output, test_name="test",
      options={
        "data_parameters": classifier_parameters,
        "plot_dir": self.plots_output, "disable_tqdm": self.disable_tqdm,
        "use_wandb": self.use_wandb,
        "save_model_per_epoch": self.save_model_per_epoch,
      },
    )
    classifier.BuildModel()
    classifier.BuildTrainer()
    classifier.Train(name=f"{self.model_output}/{self.file_name}.h5")
    report["training"] = {
      "train_loss": [float(value) for value in classifier.epoch_train_losses],
      "test_loss": [float(value) for value in classifier.epoch_test_losses],
    }
    for split in ["train", "test"]:
      report[f"roc_auc_{split}"] = self._GetROCAUC(classifier, split, batch_size)
    self._WriteYaml(f"{self.data_output}/metrics.yaml", report)
    for split in ["train", "test"]:
      print(f"ROC AUC for {split} dataset: {report[f'roc_auc_{split}']:.4f}")


  def _GetROCAUC(self, classifier, split, batch_size):
    """
    Evaluate the weighted AUC of simulation probabilities, retaining signed
    weights and giving tied scores half credit. Predict in density training space.
    """
    prediction_batch_size = max(batch_size, int(os.getenv("EVENTS_PER_BATCH", batch_size)))
    loaders = {
      key: DataLoader(f"{self.data_output}/{key}_{split}.parquet", batch_size=prediction_batch_size)
      for key in ["X", "y", "wt"]
    }
    n_rows = loaders["X"].num_rows
    labels = np.empty(n_rows, dtype=bool)
    scores = np.empty(n_rows, dtype=np.float64)
    weights = np.empty(n_rows, dtype=np.float64)
    start = 0
    try:
      for _ in range(loaders["X"].num_batches):
        X = loaders["X"].LoadNextBatch()
        end = start + len(X)
        # The paired tables have already been transformed for the density model.
        scores[start:end] = classifier.Predict(X, transform_X=False)[:, 1]
        labels[start:end] = loaders["y"].LoadNextBatch()["classifier_truth"].to_numpy()
        weights[start:end] = loaders["wt"].LoadNextBatch()["wt"].to_numpy()
        start = end
    finally:
      for loader in loaders.values():
        if loader.parquet_file is not None:
          loader.parquet_file.close()
    if n_rows == 0 or not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights)):
      raise ValueError(f"ROC AUC requires finite predictions and weights for {split}")

    # Pairwise weighted ranking also works for signed Monte Carlo weights;
    # sklearn's ROC integration requires monotonic rates, which these can violate.
    order = np.argsort(scores, kind="stable")
    scores, labels, weights = scores[order], labels[order], weights[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    positive_weights = np.add.reduceat(weights * labels, group_starts)
    negative_weights = np.add.reduceat(weights * ~labels, group_starts)
    positive_sum, negative_sum = positive_weights.sum(), negative_weights.sum()
    if positive_sum <= 0 or negative_sum <= 0:
      raise ValueError(f"ROC AUC requires positive signed weight sums in both classes for {split}")
    negatives_below = np.cumsum(negative_weights) - negative_weights
    return float(np.sum(positive_weights * (negatives_below + 0.5 * negative_weights)) / (positive_sum * negative_sum))


  def _WriteYaml(self, name, contents):
    """
    Save dataset metadata and effective training settings.
    """
    MakeDirectories(name)
    with open(name, 'w') as file:
      yaml.safe_dump(contents, file, sort_keys=False)


  def Inputs(self):
    """
    Declare the source splits, EvaluateDensity tables and architecture metadata.
    """
    inputs = [self.parameters, self.architecture,
      f"{self.density_model}_architecture.yaml"]
    for split, output_split in [(self.train_name, "train"), (self.test_name, "test")]:
      inputs += [f"{self.data_input}/{key}_{split}.parquet" for key in ["X", "Y", "wt"]]
      inputs += [f"{self.synthetic_input}/synth_{output_split}.parquet"]
    return list(dict.fromkeys(inputs))


  def Outputs(self):
    """
    Return the matched datasets, provenance, architecture and classifier weights.
    """
    outputs = [f"{self.data_output}/parameters.yaml", f"{self.data_output}/metrics.yaml",
      f"{self.model_output}/{self.file_name}_architecture.yaml"]
    for split in ["train", "test"]:
      outputs += [f"{self.data_output}/{key}_{split}.parquet" for key in ["X", "y", "wt"]]
    outputs += [f"{self.model_output}/{self.file_name}.h5"]
    return outputs
