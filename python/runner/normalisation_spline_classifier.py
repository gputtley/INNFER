import os
import pickle
import yaml

import numpy as np
import pandas as pd

from functools import partial
from scipy.interpolate import UnivariateSpline, CubicSpline

from data_processor import DataProcessor
from plotting import plot_histograms
from useful_functions import InitiateClassifierModel

class NormalisationSplineClassifier():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None
    self.cfg = None

    # other
    self.data_input = "data/"
    self.asimov_input = "data/asimov.parquet"
    self.plots_output = "plots/"
    self.file_name = None
    self.data_output = None
    self.model_input = None
    self.model_name = None
    self.parameter = None
    self.verbose = True
    self.category = None
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH"))
    self.extra_classifier_model_name = ""
    self.parameter_range = [-3.0,3.0]
    self.number_of_spline_points = 100
  
  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

    if self.extra_classifier_model_name != "":
      self.extra_classifier_model_name = f"_{self.extra_classifier_model_name}"

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
    classifier_model_name = f"{self.model_input}/{self.model_name}/{parameters['file_name']}{self.extra_classifier_model_name}"
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

    if self.verbose:
      print("- Getting predictions and building spline")

    # Make dataprocessor for predictions
    pred_df = DataProcessor(
      [self.asimov_input],
      "parquet",
      batch_size = self.batch_size,
      options = {
        "parameters" : parameters['classifier'][self.parameter],
        "sort_columns" : False,
        "wt_name" : "wt"
      },
    )

    spline_points = np.linspace(self.parameter_range[0], self.parameter_range[1], self.number_of_spline_points)

    def apply_classifier(df, func, X_columns, spline_points, parameter):
      df = df.copy()
      wt_cols = {}
      for ind, p in enumerate(spline_points):
        df[parameter] = p
        probs = func(df[X_columns])
        wt_cols[f"wt_shift_{ind}"] = probs[:, 1] / (probs[:, 0])
      wt_df = pd.DataFrame(wt_cols, index=df.index)
      df = pd.concat([df, wt_df], axis=1)
      return df

    def concatenate_df(df, X_columns,spline_points, parameter):      
      dfs = []
      for ind, p in enumerate(spline_points):
        dfs.append(df[X_columns + [f"wt_shift_{ind}", "wt"]])
        dfs[-1] = dfs[-1].rename(columns={f"wt_shift_{ind}": "wt_shift"})
        dfs[-1].loc[:, parameter] = p
      df = pd.concat(dfs, axis=0)
      return df

    def apply_shift(df):
      df["wt"] = df["wt"]*df["wt_shift"]
      return df
    

    functions_to_apply = [
      partial(
        apply_classifier, 
        func=network.Predict, 
        X_columns=parameters['classifier'][self.parameter]["X_columns"],
        spline_points=spline_points,
        parameter=self.parameter
      ),
      partial(
        concatenate_df, 
        X_columns=parameters['classifier'][self.parameter]["X_columns"],
        spline_points=spline_points,
        parameter=self.parameter
      )
    ]

    bins = list(spline_points) + [spline_points[-1]+(spline_points[-1]-spline_points[-2])]
    nom_hist, _ = pred_df.GetFull(
      method = "histogram",
      column = self.parameter,
      bins = bins,
      functions_to_apply = functions_to_apply,
    )
    shifted_hist, _ = pred_df.GetFull(
      method = "histogram",
      column = self.parameter,
      bins = bins,
      functions_to_apply = functions_to_apply + [apply_shift],
    )

    ratio = nom_hist/shifted_hist

    def weight_error(df):
      df.loc[:,"ratio_val"] = 1.0
      for ind in range(len(bins)-1):
        indices = (df[self.parameter] >= bins[ind]) & (df[self.parameter] < bins[ind+1])
        df.loc[indices,"ratio_val"] = ratio[ind]
      df.loc[:,"wt"] = (df.loc[:,"wt"] ** 2) * ((df.loc[:,"wt_shift"] - df.loc[:,"ratio_val"])**2)
      return df
    
    error_numerator_hist, _ = pred_df.GetFull(
      method = "histogram",
      column = self.parameter,
      bins = bins,
      functions_to_apply = functions_to_apply + [weight_error],
    )
    error = np.sqrt(error_numerator_hist / nom_hist**2)

    #spline = UnivariateSpline(spline_points, ratio, w=1/error, s=None)
    spline = CubicSpline(spline_points, ratio, bc_type="clamped", extrapolate=True)

    spline_name = f"{classifier_model_name}_norm_spline.pkl"
    with open(spline_name, 'wb') as f:
      pickle.dump(spline, f)

    # plot spline and points
    if self.verbose:
      print("- Plotting normalisation spline")
    plot_histograms(
      spline_points,
      [],
      [],
      error_bar_hists = [ratio],
      error_bar_hist_errs = [error],
      error_bar_names = ["Points"],
      smooth_func = spline,
      smooth_func_name = "Spline",
      name=f"{self.plots_output}/norm_spline_{self.parameter}",
      x_label=self.parameter,
      y_label="Yield ratio to before",
    )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initiate outputs
    outputs = []

    # Add output data
    outputs += [f"{self.plots_output}/norm_spline_{self.parameter}.pdf"]

    # Add output model
    outputs += [f"{self.model_input}/{self.model_name}/{self.file_name}_norm_spline.pkl"]

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
    inputs += [self.asimov_input]

    return inputs

        