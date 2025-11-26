import yaml

from data_processor import DataProcessor
from plotting import plot_histograms, plot_unrolled_2d_histogram
from useful_functions import GetParametersInModel, LoadConfig, Translate, RoundUnrolledBins

class InputPlotTraining():

  def __init__(self):
    """
    A class to preprocess the datasets and produce the data 
    parameters yaml file as well as the train, test and 
    validation datasets.
    """
    # Required input which is the location of a file
    self.cfg = None
    self.parameters = None

    # Other
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

    if self.plot_2d_unrolled:
      self._Plot2DUnrolled(specific_parameters, specific_parameters["X_columns"]+Y_cols, data_splits=["train","test"])

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Find columns
    columns = cfg["variables"]
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
        

  def _Plot1D(self, parameters, columns, n_bins=40, data_splits=["train","test"]):

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

        for col in columns:

          bins = dp.GetFull(method="bins_with_equal_spacing", bins=n_bins, functions_to_apply=functions_to_apply, column=col, ignore_quantile=0.0, ignore_discrete=True)
          bins = [(2*bins[0])-bins[1]] + list(bins) + [(2*bins[-1])-bins[-2]] + [(3*bins[-1])-(2*bins[-2])]

          hist, bins = dp.GetFull(method="histogram", bins=bins, functions_to_apply=functions_to_apply, column=col, ignore_quantile=0.0, ignore_discrete=True)

          extra_name_for_plot = f"{data_split}"
          if transform:
            extra_name_for_plot += "_transformed"
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