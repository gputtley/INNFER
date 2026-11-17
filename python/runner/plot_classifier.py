import os
import yaml
import numpy as np

from data_processor import DataProcessor
from plotting import plot_histograms_with_ratio
from useful_functions import GetVariables, LoadConfig, Translate, RoundToSF

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

      # Make 1D histogram
      if self.verbose:
        print(f"- Making distributions of reweighted samples")

      # Discrete nuisance models can have a single condition in these datasets.
      unique = pred_df.GetFull(method="unique", columns=[self.parameter], unique_threshold=self.n_plots, extra_sel="(classifier_truth == 0)")[self.parameter]
      sels = {"Inclusive": "1==1"}
      names = {"Inclusive": "inclusive"}
      if unique is not None:
        for i, value in enumerate(sorted(unique)):
          name = rf"{Translate(self.parameter, only_val=True)} = {RoundToSF(value,2)}{Translate(self.parameter, only_unit=True)}"
          sels[name] = f"({self.parameter} == {value})"
          names[name] = f"bin_{i}"
      else:
        cond_bins = pred_df.GetFull(method="bins_with_equal_stats", column=self.parameter, bins=self.n_plots, ignore_quantile=0.00, extra_sel="(classifier_truth == 0)")
        for i in range(len(cond_bins)-1):
          if cond_bins[i] == cond_bins[i+1]: continue
          upper_comparison = "<=" if i == len(cond_bins)-2 else "<"
          upper_label = r"\leq" if upper_comparison == "<=" else "<"
          name = rf"{RoundToSF(cond_bins[i],2)} $\leq$ {Translate(self.parameter, only_val=True)} ${upper_label}$ {RoundToSF(cond_bins[i+1],2)}{Translate(self.parameter, only_unit=True)}"
          sels[name] = f"(({self.parameter} >= {cond_bins[i]}) & ({self.parameter} {upper_comparison} {cond_bins[i+1]}))"
          names[name] = f"bin_{i}"

      # Plot the observables, rather than the nuisance condition itself.
      cfg = LoadConfig(self.cfg)
      columns = GetVariables(cfg, category=self.category)
      bins_by_selection = {}
      for sel_name, sel in sels.items():
        bins_by_selection[sel_name] = dict(zip(columns, pred_df.GetFull(
          method="bins_with_equal_spacing",
          columns=columns,
          bins=20,
          ignore_quantile=0.01,
          extra_sel=f"((classifier_truth == 0) & ({sel}))",
        )))

      # Fill every observable and condition histogram in one streaming pass.
      histograms = self._GetHistograms(pred_df, columns, sels, bins_by_selection)
      for col in columns:

        for sel_name, sel in sels.items():

          bins = bins_by_selection[sel_name][col]
          (nom_hist, nom_hist_uncerts), (pred_hist, pred_hist_uncerts), (target_hist, target_hist_uncerts) = histograms[sel_name][col]

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


  def _GetHistograms(self, data_processor, columns, selections, bins):
    """Accumulate nominal, reweighted and target histograms in one pass."""
    histograms = {
      name: {
        col: [[np.zeros(len(bins[name][col])-1), np.zeros(len(bins[name][col])-1)] for _ in range(3)]
        for col in columns
      }
      for name in selections
    }

    def fill_histograms(df, out, options):
      if out is None:
        out = histograms
      for name, selection in selections.items():
        selected = df if selection == "1==1" else data_processor.ApplySelection(df, extra_sel=selection)
        nominal = data_processor.ApplySelection(selected, extra_sel="classifier_truth == 0")
        target = data_processor.ApplySelection(selected, extra_sel="classifier_truth == 1")
        weights = [nominal["wt"].to_numpy(), (nominal["wt"] * nominal["wt_shift"]).to_numpy(), target["wt"].to_numpy()]
        for col in columns:
          values = [nominal[col].to_numpy(), nominal[col].to_numpy(), target[col].to_numpy()]
          for i, (value, weight) in enumerate(zip(values, weights)):
            out[name][col][i][0] += np.histogram(value, bins=bins[name][col], weights=weight)[0]
            out[name][col][i][1] += np.histogram(value, bins=bins[name][col], weights=weight**2)[0]
      return out

    data_processor.GetFull(method="custom", custom=fill_histograms)
    for selection in histograms.values():
      for hist in selection.values():
        for entry in hist:
          entry[1] = np.sqrt(entry[1])
    return histograms


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    # Initialise outputs
    outputs = []

    # Load config
    cfg = LoadConfig(self.cfg)

    # Add plots
    # Conditional-bin counts depend on the data. Declare the inclusive plots
    # which are always produced; Run also writes all available condition bins.
    loop = ["train"]
    if self.test_name is not None:
      loop.append(self.test_name)
    for data_split in loop:
      for col in GetVariables(cfg, category=self.category):
        outputs += [f"{self.plots_output}/reweighted_{col}_{data_split}_inclusive.pdf"]

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
    inputs += [f"{self.evaluate_input}/pred_train.parquet"]
    if self.test_name is not None:
      inputs += [f"{self.evaluate_input}/pred_{self.test_name}.parquet"]
  
    # Add data input
    if self.test_name is not None:
      inputs += [f"{self.data_input}/{i}_{self.test_name}.parquet" for i in ["X","y","wt"]]
    inputs += [f"{self.data_input}/{i}_train.parquet" for i in ["X","y","wt"]]

    return inputs
