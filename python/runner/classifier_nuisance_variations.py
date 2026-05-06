import yaml

import numpy as np

from functools import partial

from data_processor import DataProcessor
from plotting import plot_classsifier_nuisance_variations
from useful_functions import InitiateClassifierModel, LoadConfig, Translate
from write_parquet import WriteParquet

class ClassifierNuisanceVariations():

  def __init__(self):
    """
    Classifier nuisance variations class.
    """
    # Default values - these will be set by the configure function
    self.cfg = None
    self.parameters = None
    self.nominal_files = None
    self.up_files = None
    self.down_files = None
    self.plots_output = "plots"
    self.data_output = "data"
    self.verbose = False
    self.classifier_model = None
    self.model_input = None
    self.extra_classifier_model_name = None
    self.extra_plot_name = ""

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

    # Open the config file
    cfg = LoadConfig(self.cfg)

    # Open parameters yaml file
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build the classifier model
    if self.verbose:
      print(f"- Building classifier network for {self.classifier_model['parameter']}")
    classifier_model_name = f"{self.model_input}/{self.classifier_model['name']}{self.extra_classifier_model_name}/{parameters['file_name']}"
    with open(f"{classifier_model_name}_architecture.yaml", 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
    network = InitiateClassifierModel(
      architecture,
      self.classifier_model['file_loc'],
      options = {
        "data_parameters" : parameters['classifier'][self.classifier_model['parameter']]
      }
    )
    if self.verbose:
      print(f"- Loading the classifier model {classifier_model_name}")
    network.Load(name=f"{classifier_model_name}.h5")

    # Build data processor for nominal
    if self.verbose:
      print("- Building data processor for nominal")
    nominal_dp = DataProcessor(
      self.nominal_files,
      "parquet",
      wt_name = "wt",
      options = {}
    )

    # Make classifier application files
    def apply_classifier(df, func, X_columns, add_columns={}):
      for k,v in add_columns.items(): df.loc[:,k] = v
      probs = func(df.loc[:,X_columns])
      df["wt"] = df["wt"] * probs[:,1]/probs[:,0]
      return df.loc[:,["wt"]]
    
    if self.verbose:
      print("- Applying classifier to nominal data for up variation")

    up_wt_name = f"wt_classifier_{self.classifier_model['parameter']}_up"
    wp = WriteParquet(name=up_wt_name, data_output=self.data_output)
    nominal_dp.GetFull(
      method = None,
      functions_to_apply = [
        partial(
          apply_classifier,
          func = network.Predict,
          X_columns = parameters['classifier'][self.classifier_model['parameter']]["X_columns"],
          add_columns = {self.classifier_model['parameter']: 1.0}
        ),
        wp
      ]
    )
    wp.collect()

    if self.verbose:
      print("- Applying classifier to nominal data for down variation")

    down_wt_name = f"wt_classifier_{self.classifier_model['parameter']}_down"
    wp = WriteParquet(name=down_wt_name, data_output=self.data_output)
    nominal_dp.GetFull(
      method = None,
      functions_to_apply = [
        partial(
          apply_classifier,
          func = network.Predict,
          X_columns = parameters['classifier'][self.classifier_model['parameter']]["X_columns"],
          add_columns = {self.classifier_model['parameter']: -1.0}
        ),
        wp
      ]
    )
    wp.collect()

    # Build all data processors
    up_sim_dp = DataProcessor(
      self.up_files,
      "parquet",
      wt_name = "wt",
      options = {}
    )
    down_sim_dp = DataProcessor(
      self.down_files,
      "parquet",
      wt_name = "wt",
      options = {}
    )
    up_synth_dp = DataProcessor(
      [f for f in self.nominal_files if not f.split("/")[-1].startswith("wt")] + [f"{self.data_output}/{up_wt_name}.parquet"],
      "parquet",
      wt_name = "wt",
      options = {}
    )
    down_synth_dp = DataProcessor(
      [f for f in self.nominal_files if not f.split("/")[-1].startswith("wt")] + [f"{self.data_output}/{down_wt_name}.parquet"],
      "parquet",
      wt_name = "wt",
      options = {}
    )

    if self.verbose:
      print("- Plotting classifier variations")

    # Loop through variables
    for variable in cfg["variables"]:

      bins = nominal_dp.GetFull(
        method = "bins_with_equal_spacing",
        bins = 20,
        column = variable,
        ignore_quantile = 0.001
      )

      nom_hist, nom_hist_uncert, _ = nominal_dp.GetFull(
        method = "histogram_and_uncert",
        column = variable,
        bins = bins,
        density = True,
      )
      up_sim_hist, up_sim_hist_uncert, _ = up_sim_dp.GetFull(
        method = "histogram_and_uncert",
        column = variable,
        bins = bins,
        density = True
      )
      down_sim_hist, down_sim_hist_uncert, _ = down_sim_dp.GetFull(
        method = "histogram_and_uncert",
        column = variable,
        bins = bins,
        density = True
      )
      up_synth_hist, up_synth_hist_uncert, _ = up_synth_dp.GetFull(
        method = "histogram_and_uncert",
        column = variable,
        bins = bins,
        density = True
      )
      down_synth_hist, down_synth_hist_uncert, _ = down_synth_dp.GetFull(
        method = "histogram_and_uncert",
        column = variable,
        bins = bins,
        density = True
      )

      plot_classsifier_nuisance_variations(
        nom_hist,
        up_sim_hist,
        down_sim_hist,
        nom_hist_uncert,
        up_sim_hist_uncert,
        down_sim_hist_uncert,
        up_synth_hist,
        down_synth_hist,
        bins,
        ylabel = "Density",
        xlabel = Translate(variable),
        output_name = f"{self.plots_output}/classifier_nuisance_variations_{variable}_{self.classifier_model['parameter']}",
      )

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []

    cfg = LoadConfig(self.cfg)
    for variable in cfg["variables"]:
      outputs += [f"{self.plots_output}/classifier_nuisance_variations_{variable}_{self.classifier_model['parameter']}.pdf"]

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    inputs += [self.cfg, self.parameters]
    inputs += list((np.array(self.nominal_files).flatten()))
    inputs += list((np.array(self.up_files).flatten()))
    inputs += list((np.array(self.down_files).flatten()))

    return inputs

        