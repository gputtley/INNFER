import yaml
import numpy as np

from data_processor import DataProcessor
from plotting import plot_histograms, plot_unrolled_2d_histogram, plot_histograms_with_ratio
from useful_functions import GetVariables, GetParametersInModel, LoadConfig, Translate, RoundUnrolledBins

class InputPlotTraining():

  def __init__(self):
    """
    A class to preprocess the datasets and produce the data 
    parameters yaml file as well as the train, test and 
    validation datasets.
    """
    # Required input which is the location of a file
    self.cfg = None
    self.open_cfg = None
    self.parameters = None

    # Other
    self.category = None
    self.file_name = None
    self.parameter = None
    self.split = None
    self.model_type = "density"
    self.verbose = True
    self.data_input = "data/"
    self.plots_output = "plots/"
    self.plot_2d_unrolled = False

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

    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      cfg = LoadConfig(self.cfg)

    # Open parameters
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)    

    # Condition target
    if self.model_type == "density":
      self.condition_target = "Y"
      specific_parameters = parameters[self.model_type]
    elif self.model_type == "regression":
      self.condition_target = "y"
      specific_parameters = parameters[self.model_type][self.parameter]
    elif self.model_type == "classifier":
      self.condition_target = "y"
      specific_parameters = parameters[self.model_type][self.parameter]

    # Run 1D plot of all variables
    if self.verbose:
      print("- Making 1D distributions")

    if self.split is None:
      Y_cols = specific_parameters[f"{self.condition_target}_columns"]
    else:
      Y_cols = specific_parameters[f"split_Y_columns"][self.split]

    self._Plot1D(specific_parameters, specific_parameters["X_columns"]+Y_cols, data_splits=["train","test"])

    self._Plot1D(specific_parameters, Y_cols, data_splits=["train","test"], no_weight=True, extra_plot_name="no_weight")


    if self.plot_2d_unrolled:
      self._Plot2DUnrolled(specific_parameters, specific_parameters["X_columns"]+Y_cols, data_splits=["train","test"])

    if self.model_type == "classifier":
      self._Plot1DSplitCategory(specific_parameters, specific_parameters["X_columns"]+Y_cols, data_splits=["train","test"])


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    if self.open_cfg is not None:
      cfg = self.open_cfg
    else:
      cfg = LoadConfig(self.cfg)
      
    # Find columns
    columns = list(GetVariables(cfg, category=self.category))
    if self.model_type == "density":
      if self.split is None:
        columns += GetParametersInModel(self.file_name, cfg, only_density=True)
      else:
        columns += cfg["models"][self.file_name]["density_models"][self.split]['parameters']
    elif self.model_type == "regression":
      columns += [self.parameter]
    elif self.model_type == "classifier":
      columns += [self.parameter]

    # Add plots
    for col in columns:
      for data_split in ["train","test"]:
        outputs += [
          f"{self.plots_output}/distributions_{col}_{data_split}.pdf",
          f"{self.plots_output}/distributions_{col}_{data_split}_transformed.pdf",
        ]
    if self.plot_2d_unrolled:
      for plot_col in columns:
        for unrolled_col in columns:
          if plot_col == unrolled_col: continue
          for data_split in ["train","test"]:
            outputs += [
              f"{self.plots_output}/distributions_unrolled_2d_{plot_col}_{unrolled_col}_{data_split}.pdf",
              f"{self.plots_output}/distributions_unrolled_2d_{plot_col}_{unrolled_col}_{data_split}_transformed.pdf",
            ]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initialise inputs
    inputs = []

    # Add parameters
    inputs += [self.parameters]

    # Add data input
    inputs += [
      f"{self.data_input}/X_train.parquet", 
      f"{self.data_input}/X_test.parquet",
      f"{self.data_input}/wt_train.parquet",
      f"{self.data_input}/wt_test.parquet",
    ]
    if self.model_type == "density":
      inputs += [
        f"{self.data_input}/Y_train.parquet",
        f"{self.data_input}/Y_test.parquet",
      ]
    elif self.model_type == "regression":
      inputs += [
        f"{self.data_input}/y_train.parquet",
        f"{self.data_input}/y_test.parquet",
      ]
    elif self.model_type == "classifier":
      inputs += [
        f"{self.data_input}/y_train.parquet",
        f"{self.data_input}/y_test.parquet",
      ]

    return inputs
        

  def _Plot1D(self, parameters, columns, n_bins=40, data_splits=["train","test"], no_weight=False, extra_plot_name=""):

    for data_split in data_splits:

      dp = DataProcessor(
        [[f"{self.data_input}/X_{data_split}.parquet", f"{self.data_input}/{self.condition_target}_{data_split}.parquet", f"{self.data_input}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt" if not no_weight else None,
          "selection" : None,
          "parameters" : parameters
        }
      )

      if dp.GetFull(method="count") == 0: continue
      for transform in [False, True]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]

        hists_and_bins = dp.GetFull(method="histograms", bins=n_bins, functions_to_apply=functions_to_apply, columns=columns, ignore_quantile=0.0, ignore_discrete=True)

        for col_ind, col in enumerate(columns):

          hist = hists_and_bins[col_ind][0]
          bins = hists_and_bins[col_ind][1]

          #bins = dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, functions_to_apply=functions_to_apply, column=col, ignore_quantile=0.0, ignore_discrete=True)
          bins = [(2*bins[0])-bins[1]] + list(bins) + [(2*bins[-1])-bins[-2]] + [(3*bins[-1])-(2*bins[-2])]
          #hist, bins = dp.GetFull(method="histogram", bins=bins, functions_to_apply=functions_to_apply, column=col, ignore_quantile=0.0, ignore_discrete=True)

          hist = [0] + list(hist) + [0] + [0]

          extra_name_for_plot = f"{data_split}"
          if transform:
            extra_name_for_plot += "_transformed"
          if extra_plot_name:
            extra_name_for_plot += f"_{extra_plot_name}"
          plot_name = self.plots_output+f"/distributions_{col}_{extra_name_for_plot}"
          plot_histograms(
            bins[:-1],
            [hist],
            [None],
            title_right = "",
            name = plot_name,
            x_label = Translate(col),
            y_label = "Events",
            anchor_y_at_0 = True,
            drawstyle = "steps-mid",
          )


  def _Plot1DSplitCategory(self, parameters, columns, n_bins=40, data_splits=["train","test"]):

    for data_split in data_splits:

      dp = DataProcessor(
        [[f"{self.data_input}/X_{data_split}.parquet", f"{self.data_input}/{self.condition_target}_{data_split}.parquet", f"{self.data_input}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : None,
          "parameters" : parameters
        }
      )

      if dp.GetFull(method="count") == 0: continue
      for transform in [False, True]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]

        columns_no_truth = [col for col in columns if col != "classifier_truth"]
        hists_and_bins_0 = dp.GetFull(method="histograms", bins=n_bins, functions_to_apply=functions_to_apply, columns=columns_no_truth, ignore_quantile=0.0, ignore_discrete=True, extra_sel="(classifier_truth == 0)")
        # get bins from output
        bins = {col: hists_and_bins_0[col_ind][1] for col_ind, col in enumerate(columns_no_truth)}
        hists_and_bins_1 = dp.GetFull(method="histograms", bins=bins, functions_to_apply=functions_to_apply, columns=columns_no_truth, ignore_quantile=0.0, ignore_discrete=True, extra_sel="(classifier_truth == 1)")
        hist_and_bins_0_negative = dp.GetFull(method="histograms", bins=bins, functions_to_apply=functions_to_apply, columns=columns_no_truth, ignore_quantile=0.0, ignore_discrete=True, extra_sel=f"((classifier_truth == 0) & ({self.parameter}<0))")
        hist_and_bins_0_positive = dp.GetFull(method="histograms", bins=bins, functions_to_apply=functions_to_apply, columns=columns_no_truth, ignore_quantile=0.0, ignore_discrete=True, extra_sel=f"((classifier_truth == 0) & ({self.parameter}>=0))")


        for col_ind, col in enumerate(columns_no_truth):

          hist_0 = hists_and_bins_0[col_ind][0]
          bins = hists_and_bins_0[col_ind][1]
          hist_1 = hists_and_bins_1[col_ind][0]
          if hist_and_bins_0_positive is not None:
            hist_0_positive = hist_and_bins_0_positive[col_ind][0]
          if hist_and_bins_0_negative is not None:
            hist_0_negative = hist_and_bins_0_negative[col_ind][0]
          
          bins = [(2*bins[0])-bins[1]] + list(bins) + [(2*bins[-1])-bins[-2]] + [(3*bins[-1])-(2*bins[-2])]

          hist_0 = [0] + list(hist_0) + [0] + [0]
          hist_1 = [0] + list(hist_1) + [0] + [0]
          if hist_and_bins_0_negative is not None:
            hist_0_negative = [0] + list(hist_0_negative) + [0] + [0]
          if hist_and_bins_0_positive is not None:
            hist_0_positive = [0] + list(hist_0_positive) + [0] + [0]

          extra_name_for_plot = f"{data_split}"
          if transform:
            extra_name_for_plot += "_transformed"
          plot_name = self.plots_output+f"/distributions_truth_split_{col}_{extra_name_for_plot}"
          plot_histograms(
            bins[:-1],
            [hist_0, hist_1],
            ["Shifted", "Nominal"],
            title_right = "",
            name = plot_name,
            x_label = Translate(col),
            y_label = "Events",
            anchor_y_at_0 = True,
            drawstyle = "steps-mid",
          )

          plot_name_shifted = self.plots_output+f"/distributions_truth_split_pos_neg_{col}_{extra_name_for_plot}"

          hists_to_plots = []
          hists_unnorm_to_plot = []
          names = []
          if hist_and_bins_0_negative is not None:
            hists_to_plots.append(hist_0_negative/np.sum(hist_0_negative))
            hists_unnorm_to_plot.append(hist_0_negative)
            names.append("Shifted Negative")
          if hist_and_bins_0_positive is not None:
            hists_to_plots.append(hist_0_positive/np.sum(hist_0_positive))
            hists_unnorm_to_plot.append(hist_0_positive)
            names.append("Shifted Positive")
          hists_to_plots.append(hist_1/np.sum(hist_1))
          hists_unnorm_to_plot.append(hist_1)
          names.append("Nominal")

          if col != self.parameter:

            plot_histograms(
              bins[:-1],
              hists_to_plots,
              names,
              title_right = "",
              name = plot_name_shifted,
              x_label = Translate(col),
              y_label = "Density",
              anchor_y_at_0 = True,
              drawstyle = "steps-mid",
            )

            ratio_hists_to_plot = []
            ratio_uncertainties_to_plot = []
            ratio_names = []
            if hist_and_bins_0_negative is not None:
              ratio_hists_to_plot.append([hist_0_negative/np.sum(hist_0_negative), hist_1/np.sum(hist_1)])
              ratio_uncertainties_to_plot.append([np.zeros_like(hist_0_negative), np.zeros_like(hist_1)])
              ratio_names.append(["Shifted Negative", "Nominal"])
            if hist_and_bins_0_positive is not None:
              ratio_hists_to_plot.append([hist_0_positive/np.sum(hist_0_positive), hist_1/np.sum(hist_1)])
              ratio_uncertainties_to_plot.append([np.zeros_like(hist_0_positive), np.zeros_like(hist_1)])
              ratio_names.append(["Shifted Positive", "Nominal"])
            
            plot_histograms_with_ratio(
              ratio_hists_to_plot,
              ratio_uncertainties_to_plot,
              ratio_names,
              bins,
              xlabel = Translate(col),
              ylabel = "Density",
              anchor_y_at_0 = True,
              ratio_range = [0.9,1.1],
              first_ratio = True,
              name = f"{plot_name_shifted}_ratio"
            )


          else:

            plot_histograms(
              bins[:-1],
              hists_unnorm_to_plot,
              names,
              title_right = "",
              name = plot_name_shifted,
              x_label = Translate(col),
              y_label = "Events",
              anchor_y_at_0 = True,
              drawstyle = "steps-mid",
            )      


  def _Plot2DUnrolled(self, parameters, columns, n_bins=10, n_unrolled_bins=5, data_splits=["train","test"]):

    for data_split in data_splits:

      dp = DataProcessor(
        [[f"{self.data_input}/X_{data_split}.parquet", f"{self.data_input}/{self.condition_target}_{data_split}.parquet", f"{self.data_input}/wt_{data_split}.parquet"]], 
        "parquet",
        options = {
          "wt_name" : "wt",
          "selection" : None,
          "parameters" : parameters
        }
      )

      if dp.GetFull(method="count") == 0: continue
      for transform in [False, True]:
        functions_to_apply = []
        if not transform:
          functions_to_apply = ["untransform"]

        for plot_col_ind, plot_col in enumerate(columns):

          # Get bins for plot_col
          plot_col_bins = dp.GetFull(
            method = "bins_with_equal_spacing", 
            functions_to_apply = functions_to_apply,
            bins = n_bins,
            column = plot_col,
          )

          for unrolled_col_ind, unrolled_col in enumerate(columns):

            # Skip if the same column
            if plot_col == unrolled_col: continue

            # Get bins for plot_col
            unrolled_col_bins = dp.GetFull(
              method = "bins_with_equal_stats", 
              functions_to_apply = functions_to_apply,
              bins = n_unrolled_bins,
              column = unrolled_col,
            )
            unrolled_col_bins = RoundUnrolledBins(unrolled_col_bins)

            # Make histograms
            hist, hist_uncert, bins = dp.GetFull(
              method = "histogram_2d_and_uncert",
              functions_to_apply = functions_to_apply,
              bins = [unrolled_col_bins, plot_col_bins],
              column = [unrolled_col, plot_col],
              )

            extra_name_for_plot = f"{data_split}"
            if transform:
              extra_name_for_plot += "_transformed"
            plot_unrolled_2d_histogram(
              {Translate(self.file_name) : hist},
              bins[1],
              bins[0], 
              Translate(unrolled_col),
              xlabel=Translate(plot_col),
              ylabel="Events",
              name=f"{self.plots_output}/distributions_unrolled_2d_{plot_col}_{unrolled_col}_{extra_name_for_plot}", 
              hists_errors={Translate(self.file_name) : hist_uncert}, 
            )