import xgboost as xgb

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer

from data_processor import DataProcessor
from plotting import plot_histograms_with_ratio
from useful_functions import CustomHistogram,LoadConfig,Translate

class PostFitTruthComparison():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.cfg = None
    self.truth_input = None
    self.best_fit_input = None
    self.fit_data_input = None

    self.extra_plot_name = ""
    self.plots_output = "plots/"
    self.verbose = True

    self.best_fit_efficiency = 1.0


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
    # Code the run the class
    if self.verbose:
      print("- Loading in config")
    cfg = LoadConfig(self.cfg)


    # Make data processors
    truth_dps = {}
    best_fit_dps = {}
    fit_data_dps = {}
    for k in self.truth_input.keys():
      if self.verbose:
        print(f"- Making data processor for truth input {k}")
      truth_dps[k] = DataProcessor(
        [self.truth_input[k]],
        "parquet",
        wt_name = "wt",
        options = {
          "check_wt" : True,
        }
      )
    for k in self.best_fit_input.keys():
      if self.verbose:
        print(f"- Making data processor for best fit input {k}")
      best_fit_dps[k] = DataProcessor(
        [self.best_fit_input[k]],
        "parquet",
        wt_name = "wt",
        options = {
          "check_wt" : True,
        }
      )
    for k in self.fit_data_input.keys():
      if self.verbose:
        print(f"- Making data processor for fit data input {k}")
      fit_data_dps[k] = DataProcessor(
        [self.fit_data_input[k]],
        "parquet",
        wt_name = "wt",
        options = {
          "check_wt" : True,
        }
      )


    # Load full data for truth and best fit and fit_data
    first = True
    for k, v in truth_dps.items():
      if self.verbose:
        print(f"- Loading full data for truth input {k}")
      tmp = v.GetFull(method="dataset")
      if first:
        truth_df = tmp.copy()
        first = False
      else:
        truth_df = pd.concat([truth_df, tmp], ignore_index=True)

    first = True
    for k, v in best_fit_dps.items():
      if self.verbose:
        print(f"- Loading full data for best fit input {k}")
      tmp = v.GetFull(method="dataset")
      if first:
        best_fit_df = tmp.copy()
        first = False
      else:
        best_fit_df = pd.concat([best_fit_df, tmp], ignore_index=True)

    first = True
    for k, v in fit_data_dps.items():
      if self.verbose:
        print(f"- Loading full data for fit data input {k}")
      tmp = v.GetFull(method="dataset")
      if first:
        fit_data_df = tmp.copy()
        first = False
      else:
        fit_data_df = pd.concat([fit_data_df, tmp], ignore_index=True)

    # Separate with a BDT
    truth_df["is_best_fit"] = 0
    best_fit_df["is_best_fit"] = 1
    combined_df = pd.concat([truth_df, best_fit_df], ignore_index=True)
    """
    train_df, test_df = train_test_split(combined_df, test_size=0.5, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.005,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_lambda=10,
        random_state=42,
    )
    model.fit(
      train_df.drop(columns=["is_best_fit", "wt"]),
      train_df["is_best_fit"],
      sample_weight = train_df["wt"]
    )
    train_df["bdt_score"] = model.predict_proba(train_df.drop(columns=["is_best_fit", "wt"]))[:,1]
    test_df["bdt_score"] = model.predict_proba(test_df.drop(columns=["is_best_fit", "wt"]))[:,1]
    train_roc_auc = roc_auc_score(train_df["is_best_fit"], train_df["bdt_score"], sample_weight=train_df["wt"])
    test_roc_auc = roc_auc_score(test_df["is_best_fit"], test_df["bdt_score"], sample_weight=test_df["wt"])
    print(f"Train ROC AUC: {train_roc_auc}")
    print(f"Test ROC AUC: {test_roc_auc}")

    """

    X = combined_df.drop(columns=["is_best_fit", "wt"])
    y = combined_df["is_best_fit"]
    w = combined_df["wt"]
    cv = StratifiedKFold(
      n_splits=2,
      shuffle=True,
      random_state=42
    )
    base_model = xgb.XGBClassifier(
      objective="binary:logistic",
      eval_metric="auc",
      random_state=42,
      n_jobs=1,            # reproducibility
      tree_method="hist"
    )
    param_grid = {
      "n_estimators": [50, 100],
      "max_depth": [2, 3],
      "learning_rate": [0.005, 0.01],
    }
    roc_scorer = make_scorer(
      roc_auc_score,
      #needs_proba=True,
      response_method='predict_proba',
    )
    #grid = GridSearchCV(
    #    estimator=base_model,
    #    param_grid=param_grid,
    #    scoring=roc_scorer,
    #    cv=cv,
    #    verbose=2,
    #    refit=True
    #)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=1,
        scoring=roc_scorer,
        cv=cv,
        random_state=42,
        verbose=2,
        n_jobs=1
    )

    search.fit(X, y, sample_weight=w)

    best_model = search.best_estimator_
    combined_df["bdt_score"] = best_model.predict_proba(X)[:, 1]
    fit_data_df["bdt_score"] = best_model.predict_proba(
        fit_data_df.drop(columns=["wt"])
    )[:, 1]
    combined_roc_auc = roc_auc_score(combined_df["is_best_fit"], combined_df["bdt_score"], sample_weight=combined_df["wt"])
    print(f"Combined ROC AUC: {combined_roc_auc}")



    
    # Draw histogram of BDT score
    #bins = np.quantile(combined_df["bdt_score"], np.linspace(0.01,0.99,6))

    truth_hist, truth_hist_uncert, bins = CustomHistogram(
      combined_df.loc[combined_df["is_best_fit"]==0, "bdt_score"],
      weights=combined_df.loc[combined_df["is_best_fit"]==0, "wt"],
      bins=5,
      ignore_quantile=0.005,
      add_uncert = True,
    )
    best_fit_hist, best_fit_hist_uncert, _ = CustomHistogram(
      combined_df.loc[combined_df["is_best_fit"]==1, "bdt_score"],
      weights=combined_df.loc[combined_df["is_best_fit"]==1, "wt"],
      bins=bins,
      add_uncert = True,
    )
    fit_data_hist, fit_data_hist_uncert, _ = CustomHistogram(
      fit_data_df["bdt_score"],
      weights=fit_data_df["wt"],
      bins=bins,
      add_uncert = True,
    )

    ratio = best_fit_hist / truth_hist
    data_ratio = fit_data_hist / truth_hist
    max_ratio = np.max(ratio)
    min_ratio = np.min(ratio)
    max_data_ratio = np.max(data_ratio)
    min_data_ratio = np.min(data_ratio)
    max_ratio = np.max([max_ratio, max_data_ratio])
    min_ratio = np.min([min_ratio, min_data_ratio])
    max_ratio_diff = 1.1*np.max([np.abs(1 - max_ratio), np.abs(1 - min_ratio)])


    if self.extra_plot_name != "":
      self.extra_plot_name = "_" + self.extra_plot_name

    plot_histograms_with_ratio(
      [[fit_data_hist, truth_hist], [best_fit_hist, truth_hist]],
      [[fit_data_hist_uncert, truth_hist_uncert], [best_fit_hist_uncert, truth_hist_uncert]],
      [["Data", "Truth"], ["Best Fit", "Truth"]],
      bins,
      xlabel = "BDT Score",
      ylabel = "Events",
      name = f"{self.plots_output}/postfit_truth_comparison{self.extra_plot_name}",
      ratio_range = [1 - max_ratio_diff, 1 + max_ratio_diff],
      draw_error_bars_first = True,
      ignore_error_in_ratio = True,
      first_ratio=True,
      draw_error_bar_caps=False,
    )

    # Find quantile where the best fit efficiency is reached
    for best_fit_efficiency in [0.25, 0.5, 0.75, 1.0]:
      threshold = np.quantile(
        combined_df.loc[combined_df["is_best_fit"]==1, "bdt_score"],
        1 - best_fit_efficiency
      )

      # Apply threshold to data
      fit_data_df_efficiency_cut = fit_data_df.loc[fit_data_df["bdt_score"] >= threshold]
      combined_df_efficiency_cut = combined_df.loc[combined_df["bdt_score"] >= threshold]
      truth_df_efficiency_cut = combined_df_efficiency_cut.loc[combined_df_efficiency_cut["is_best_fit"]==0]
      best_fit_df_efficiency_cut = combined_df_efficiency_cut.loc[combined_df_efficiency_cut["is_best_fit"]==1]
      #truth_df_efficiency_cut = truth_df.loc[truth_df["bdt_score"] >= threshold]
      #best_fit_df_efficiency_cut = best_fit_df.loc[best_fit_df["bdt_score"] >= threshold]

      for col in cfg["variables"]:

        truth_hist, truth_uncert, bins = CustomHistogram(
          truth_df_efficiency_cut[col],
          weights=truth_df_efficiency_cut["wt"],
          bins=10,
          ignore_quantile=0.005,
          add_uncert = True,
        )
        best_fit_hist, best_fit_uncert, _ = CustomHistogram(
          best_fit_df_efficiency_cut[col],
          weights=best_fit_df_efficiency_cut["wt"],
          bins=bins,
          add_uncert = True,
        )
        fit_data_hist, fit_data_uncert, _ = CustomHistogram(
          fit_data_df_efficiency_cut[col],
          weights=fit_data_df_efficiency_cut["wt"],
          bins=bins,
          add_uncert = True,
        )

        # Get ratio range
        ratio = best_fit_hist / truth_hist
        data_ratio = fit_data_hist / truth_hist
        max_ratio = np.max(ratio)
        min_ratio = np.min(ratio)
        max_data_ratio = np.max(data_ratio)
        min_data_ratio = np.min(data_ratio)
        max_ratio = np.max([max_ratio, max_data_ratio])
        min_ratio = np.min([min_ratio, min_data_ratio])
        max_ratio_diff = 1.1*np.max([np.abs(1 - max_ratio), np.abs(1 - min_ratio)])

        plot_histograms_with_ratio(
          [[fit_data_hist, truth_hist], [best_fit_hist, truth_hist]],
          [[fit_data_uncert, truth_uncert], [best_fit_uncert, truth_uncert]],
          [["Data", "Truth"], ["Best Fit", "Truth"]],
          bins,
          xlabel = Translate(col),
          ylabel = "Events",
          name = f"{self.plots_output}/postfit_truth_comparison_{col}_eff{str(best_fit_efficiency).replace('.','p')}{self.extra_plot_name}",
          ratio_range = [1 - max_ratio_diff, 1 + max_ratio_diff],
          draw_error_bars_first = True,
          ignore_error_in_ratio = True,
          first_ratio=True,
          draw_error_bar_caps=False,
          axis_text = f"Best Fit Efficiency {best_fit_efficiency*100:.2f}% Cut"
        )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []

    # Add plots
    outputs += [f"{self.plots_output}/postfit_truth_comparison{self.extra_plot_name}.pdf"]

    # Load config to get variables
    cfg = LoadConfig(self.cfg)
    for col in cfg["variables"]:
      for best_fit_efficiency in [0.25, 0.5, 0.75, 1.0]:
        outputs += [f"{self.plots_output}/postfit_truth_comparison_{col}_eff{str(best_fit_efficiency).replace('.','p')}{self.extra_plot_name}.pdf"]

      #outputs += [f"{self.plots_output}/postfit_truth_comparison_{col}{self.extra_plot_name}.pdf"]

    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add files
    for k in self.truth_input.keys():
      inputs += self.truth_input[k]
    for k in self.best_fit_input.keys():
      inputs += self.best_fit_input[k]
    for k in self.fit_data_input.keys():
      inputs += self.fit_data_input[k]

    return inputs

        