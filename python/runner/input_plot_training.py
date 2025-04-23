import yaml

from data_processor import DataProcessor
from plotting import plot_histograms
from useful_functions import GetParametersInModel, LoadConfig

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
    self.model_type = "density"
    self.verbose = True
    self.data_input = "data/"
    self.plots_output = "plots/"

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

    # Run 1D plot of all variables
    if self.verbose:
      print("- Making 1D distributions")
      self._Plot1D(specific_parameters, specific_parameters["X_columns"]+specific_parameters[f"{self.condition_target}_columns"], data_splits=["train","test"])


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
      columns += GetParametersInModel(self.file_name, cfg, only_density=True)
    elif self.model_type == "regression":
      columns += [self.parameter]

    # Add plots
    for col in columns:
      for data_split in ["train","test"]:
        outputs += [
          f"{self.plots_output}/distributions_{col}_{data_split}.pdf",
          f"{self.plots_output}/distributions_{col}_{data_split}_transformed.pdf",
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
            x_label = col,
            y_label = "Events",
            anchor_y_at_0 = True,
            drawstyle = "steps-mid",
          )