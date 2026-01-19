import copy
import os
import pickle
import yaml

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from functools import partial
from scipy.interpolate import CubicSpline

from data_processor import DataProcessor
from plotting import plot_histograms
from useful_functions import InitiateClassifierModel, MakeDirectories

class EvaluateClassifier():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None

    # other
    self.data_input = "data/"
    self.plots_output = "plots/"
    self.file_name = None
    self.data_output = None
    self.model_input = None
    self.model_name = None
    self.parameter = None
    self.verbose = True
    self.test_name = "test"
    self.model_type = "FCNN"
    self.spline_from_asimov = False
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    self.get_auc = True


  def _WriteDataset(self, df, file_name):

    file_path = f"{self.data_output}/{file_name}"
    MakeDirectories(file_path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.isfile(file_path):
      combined_table = pa.concat_tables([pq.read_table(file_path), table])
      pq.write_table(combined_table, file_path, compression='snappy')
    else:
      pq.write_table(table, file_path, compression='snappy')

    return df

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    # Open parameters
    if self.verbose:
      print("- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Load the model in
    if self.verbose:
      print("- Building the model")
    classifier_model_name = f"{self.model_input}/{self.model_name}/{parameters['file_name']}"
    with open(f"{classifier_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
    network = InitiateClassifierModel(
      architecture,
      self.data_input,
      options = {
        "data_parameters" : parameters['classifier'][self.parameter]
      }
    )  
    network.BuildModel()
    network.BuildTrainer()
    network.Load(name=f"{classifier_model_name}.h5")

    # Make y from train and test
    loop = ["train"]
    if self.test_name is not None:
      loop.append(self.test_name)

    performance_metrics = {}

    for data_split in loop:
      if self.verbose:
        print(f"- Getting to the predictions for the {data_split} dataset")

      pred_df = DataProcessor(
        [[f"{parameters['classifier'][self.parameter]['file_loc']}/{i}_{data_split}.parquet" for i in ["X","y"]]],
        "parquet",
        batch_size = self.batch_size,
        options = {
          "parameters" : parameters['classifier'][self.parameter],
        },
      )

      def apply_classifier(df, func, X_columns):
        df.loc[:,"wt_shift"] = 1.0
        probs = func(df.loc[:,X_columns])
        inds = (df["classifier_truth"] == 0)
        df.loc[inds, "wt_shift"] = probs[inds,1] / probs[inds, 0]
        df.loc[:, "probs"] = probs[:,1]
        return df.loc[:,["wt_shift","probs"]]

      pred_name = f"{self.data_output}/pred_{data_split}.parquet"
      if os.path.isfile(pred_name): os.system(f"rm {pred_name}")

      pred_df.GetFull(
        method = None,
        functions_to_apply = [
          "untransform",
          partial(
            apply_classifier, 
            func=network.Predict, 
            X_columns=parameters['classifier'][self.parameter]["X_columns"],
          ),
          partial(self._WriteDataset,file_name=f"pred_{data_split}.parquet")
        ]
      )


      # Get ROC AUC from predictions
      if self.get_auc:

        if self.verbose:
          print("- Getting ROC AUC")

        pred_df = DataProcessor(
          [f"{self.data_output}/pred_{data_split}.parquet", f"{parameters['classifier'][self.parameter]['file_loc']}/y_{data_split}.parquet", f"{parameters['classifier'][self.parameter]['file_loc']}/wt_{data_split}.parquet"],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : parameters['classifier'][self.parameter],
          }
        )
        df = pred_df.GetFull(method="dataset")
        df = df[(df["wt"] > 0)]
        auc = roc_auc_score(df["classifier_truth"], df["probs"], sample_weight=df["wt"])
        performance_metrics[f"roc_auc_{data_split}"] = float(auc)
        print(f"ROC AUC for {data_split} dataset: {auc:.4f}")



      # Get normalisation spline
      if data_split == "train":

        if self.verbose:
          print("- Getting normalisation spline")

        reweight_df = DataProcessor(
          [f"{parameters['classifier'][self.parameter]['file_loc']}/X_{data_split}.parquet", f"{parameters['classifier'][self.parameter]['file_loc']}/y_{data_split}.parquet", f"{parameters['classifier'][self.parameter]['file_loc']}/wt_{data_split}.parquet", pred_name],
          "parquet",
          wt_name = "wt",
          options = {
            "parameters" : parameters['classifier'][self.parameter],
          }
        )

        # Get number of effective events
        eff_events = reweight_df.GetFull(
          method = "n_eff"
        )
        events_per_bin = 10000
        bins = min(int(np.ceil(eff_events/events_per_bin)),10)

        if self.verbose:
          print(f"- Number of bins: {bins}")

        nom_hist, bins = reweight_df.GetFull(
          method = "histogram",
          column = self.parameter,
          bins = bins,
          extra_sel = "(classifier_truth == 0)",
          ignore_quantile=0.0,
          functions_to_apply=["untransform"]
        )

        def shift(df):
          df.loc[:,"wt"] *= df.loc[:,"wt_shift"]
          return df

        shifted_hist, _ = reweight_df.GetFull(
          method = "histogram",
          bins = bins,
          column = self.parameter,
          functions_to_apply = [shift, "untransform"],
          extra_sel = "(classifier_truth == 0)",
        )

        ratio = nom_hist/shifted_hist
        bin_centers = (bins[:-1] + bins[1:]) / 2
        spline = CubicSpline(bin_centers, ratio, bc_type="clamped", extrapolate=True)
        spline_name = f"{classifier_model_name}_norm_spline.pkl"
        with open(spline_name, 'wb') as f:
          pickle.dump(spline, f)

        # plot spline and points
        if self.verbose:
          print("- Plotting normalisation spline")
        plot_histograms(
          bin_centers,
          [],
          [],
          error_bar_hists = [ratio],
          error_bar_hist_errs = [np.zeros(len(ratio))],
          error_bar_names = ["Points"],
          smooth_func = spline,
          smooth_func_name = "Spline",
          name=f"{self.plots_output}/norm_spline_{self.parameter}",
          x_label=self.parameter,
          y_label="Yield ratio to before",
        )

    # Write out performance metrics
    if self.verbose:
      print("- Writing out performance metrics")
    metrics_name = f"{self.data_output}/performance_metrics.yaml"
    MakeDirectories(metrics_name)
    with open(metrics_name, 'w') as yaml_file:
      yaml.dump(performance_metrics, yaml_file)


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Add output data
    outputs += [f"{self.data_output}/pred_train.parquet"]
    if self.test_name is not None:
      outputs += [f"{self.data_output}/pred_{self.test_name}.parquet"]

    # Add normalisation spline
    outputs += [f"{self.model_input}/{self.model_name}/{self.file_name}_norm_spline.pkl"]

    # Add performance metrics
    outputs += [f"{self.data_output}/performance_metrics.yaml"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initiate inputs
    inputs = []

    # Add parameters
    inputs += [self.parameters]

    # Add model
    inputs += [f"{self.model_input}/{self.model_name}/{self.file_name}_architecture.yaml"]
    inputs += [f"{self.model_input}/{self.model_name}/{self.file_name}.h5"]

    # Add data
    inputs += [
      f"{self.data_input}/X_train.parquet", 
      f"{self.data_input}/wt_train.parquet",
    ]
    if self.test_name is not None:
      inputs += [
        f"{self.data_input}/X_{self.test_name}.parquet",
        f"{self.data_input}/wt_{self.test_name}.parquet",
      ]

    return inputs

        