import os
import pickle
import yaml

from functools import partial

from data_processor import DataProcessor
from plotting import plot_histograms_with_ratio
from useful_functions import LoadConfig, Translate, RoundToSF

class PlotClassifier():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None
    self.cfg = None

    # other
    self.n_plots = 10
    self.data_input = "data/"
    self.evaluate_input = "data/"
    self.plots_output = "plots/"
    self.model_name = None
    self.parameter = None
    self.verbose = True
    self.test_name = "test"
    self.model_input = None
    self.batch_size = int(os.getenv("EVENTS_PER_BATCH"))

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

    # Make y from train and test
    loop = ["train"]
    if self.test_name is not None:
      loop.append(self.test_name)

    for data_split in loop:

      if self.verbose:
        print(f"- Building data processors for the {data_split} dataset")

      nominal_files = [f"{self.data_input}/{i}_{data_split}.parquet" for i in ["X","y"]]
      pred_file = f"{self.evaluate_input}/pred_{data_split}.parquet"
      wt_norm_file = f"{self.data_input}/wt_{data_split}.parquet"

      pred_df = DataProcessor(
        [nominal_files + [pred_file] + [wt_norm_file]],
        "parquet",
        wt_name = "wt",
        batch_size = self.batch_size,
        options = {
          "parameters" : parameters['classifier'][self.parameter],
          "functions" : ["untransform"],
        }
      )

      # Load spline
      classifier_model_name = f"{self.model_input}/{self.model_name}/{parameters['file_name']}"
      spline_name = f"{classifier_model_name}_norm_spline.pkl"
      if os.path.isfile(spline_name):
        with open(spline_name, 'rb') as f:
          spline = pickle.load(f)

      # Make 1D histogram
      if self.verbose:
        print(f"- Making distributions of reweighted samples")

      for col in parameters["classifier"][self.parameter]["X_columns"]:

        # Get bins of conditional variables
        cond_bins = pred_df.GetFull(method="bins_with_equal_stats", column=self.parameter, bins=self.n_plots, ignore_quantile=0.00, extra_sel="(classifier_truth == 0)")

        sels = {"Inclusive": "1==1"}
        names = {"Inclusive": "inclusive"}
        for i in range(len(cond_bins)-1):
          name = rf"{RoundToSF(cond_bins[i],2)} $\leq$ {Translate(self.parameter)} < {RoundToSF(cond_bins[i+1],2)}"
          sels[name] = f"(({self.parameter} >= {cond_bins[i]}) & ({self.parameter} < {cond_bins[i+1]}))"
          names[name] = f"bin_{i}"

        for sel_name, sel in sels.items():

          # get nominal sums
          bins = pred_df.GetFull(method="bins_with_equal_spacing", column=col, bins=20, ignore_quantile=0.01, extra_sel=f"((classifier_truth == 0) & ({sel}))")
          nom_hist, nom_hist_uncerts, _ = pred_df.GetFull(method="histogram_and_uncert", column=col, bins=bins, extra_sel=f"((classifier_truth == 0) & ({sel}))")

          # get shift applied histogram
          def change_weight(df, spline):
            spline_vals = spline(df.loc[:,self.parameter])
            df.loc[:,"wt"] *= df.loc[:,"wt_shift"] * spline_vals
            return df

          pred_hist, pred_hist_uncerts, _ = pred_df.GetFull(method="histogram_and_uncert", column=col, bins=bins, functions_to_apply=[partial(change_weight, spline=spline)], extra_sel=f"((classifier_truth == 0) & ({sel}))")
          target_hist, target_hist_uncerts, _ = pred_df.GetFull(method="histogram_and_uncert", column=col, bins=bins, extra_sel=f"((classifier_truth == 1) & ({sel}))")

          plot_histograms_with_ratio(
            [[pred_hist, target_hist], [nom_hist, target_hist]],
            [[pred_hist_uncerts, target_hist_uncerts], [nom_hist_uncerts, target_hist_uncerts]],
            [["Reweighted", "Target"], ["Original","Target"]],
            bins,
            xlabel = col,
            ylabel="Events",
            name=f"{self.plots_output}/reweighted_{col}_{data_split}_{names[sel_name]}",      
            ratio_range = [0.9,1.1],
            first_ratio = True,
            axis_text=sel_name
          )


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add plots
    for data_split in ["train", self.test_name]:
      for col in cfg["variables"]:
        outputs += [f"{self.plots_output}/reweighted_{col}_{data_split}_inclusive.pdf"]
        for i in range(self.n_plots):
          outputs += [f"{self.plots_output}/reweighted_{col}_{data_split}_bin_{i}.pdf"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    # Initialise inputs
    inputs = []

    # Add parameters
    inputs += [self.parameters]

    # Add pred input
    inputs += [
      f"{self.evaluate_input}/pred_train.parquet", 
      f"{self.evaluate_input}/pred_{self.test_name}.parquet",
    ]
  
    # Add data input
    inputs += [f"{self.data_input}/{i}_{self.test_name}.parquet" for i in ["X","y","wt"]]
    inputs += [f"{self.data_input}/{i}_train.parquet" for i in ["X","y","wt"]]

    return inputs

        