import optuna
import os
import wandb
import yaml

from random_word import RandomWords

from train import Train
from performance_metrics import PerformanceMetrics
from useful_functions import FindKeysAndValuesInDictionaries, MakeDictionaryEntry, GetDictionaryEntryFromYaml, MakeDirectories

class BayesianHyperparameterTuning():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.parameters = None
    self.tune_architecture = None

    # other
    self.best_model_output = None
    self.metric = ""
    self.use_wandb = False
    self.wandb_project_name = "innfer"
    self.wandb_submit_name = "innfer"
    self.verbose = True
    self.disable_tqdm = False
    self.data_output = "data/"
    self.n_trials = 10

    self.objective_ind = 0
    self.tune_architecture_name = None

  def _TrainAndPerfomanceMetric(self):
    """
    Run the code utilising the worker classes
    """
    # Setup wandb
    if self.use_wandb:

      if self.verbose:
        print("- Initialising wandb")

      with open(self.tune_architecture_name, 'r') as yaml_file:
        architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
      r = RandomWords()
      wandb.init(project=self.wandb_project_name, name=f"{self.wandb_submit_name}_{r.get_random_word()}", config=architecture)

    # Train models
    if self.verbose:
      print("- Training the models")
    t = self._SetupTrain()
    t.Run()

    # Get performance metrics
    if self.verbose:
      print("- Getting the performance metrics")
    pf = self._SetupPerformanceMetrics()
    pf.Run()

    # Write performance metrics to wandb
    if self.use_wandb:
      if self.verbose:
        print("- Writing performance metrics to wandb")
      metric_name = f"{self.data_output}/metrics_{self.objective_ind}.yaml"
      with open(metric_name, 'r') as yaml_file:
        metric = yaml.load(yaml_file, Loader=yaml.FullLoader)
      wandb.log(metric)
      wandb.finish()

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

    if self.metric.split(",")[1] == "min":
      direction = "minimize"
    elif self.metric.split(",")[1] == "max":
      direction = "maximize"

    study = optuna.create_study(direction=direction)
    study.optimize(self._Objective, n_trials=self.n_trials)

    # Get best trial
    best_trial = study.best_trial.number

    # Move best model and architecture to correct directory
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    model_output_name = f"{self.best_model_output}/{parameters['file_name']}.h5"
    model_architecture_name = f"{self.best_model_output}/{parameters['file_name']}_architecture.yaml"
    best_model_name = f"{self.data_output}/{parameters['file_name']}_{best_trial}.h5"
    best_architecture_name = f"{self.data_output}/{parameters['file_name']}_{best_trial}_architecture.yaml"


    if self.verbose:
      print("- Best model is:")
      print(f"  {best_model_name}")
      print(f"  {best_architecture_name}")

    MakeDirectories(model_output_name)
    os.system(f"cp {best_model_name} {model_output_name}")
    os.system(f"cp {best_architecture_name} {model_architecture_name}")


  def _Objective(self, trial):

    # Open tuning architecture
    with open(self.tune_architecture, 'r') as yaml_file:
      config = yaml.load(yaml_file, Loader=yaml.FullLoader)  

    # Make trial parameter
    keys, vals = FindKeysAndValuesInDictionaries(config, keys=[], results_keys=[], results_vals=[])
    run_vals = []
    for ind, val in enumerate(vals):
      if isinstance(val, list):
        if isinstance(val[0], float):
          run_vals.append(trial.suggest_float(":".join(keys[ind]), val[0], val[1]))
        elif isinstance(val[0], int):
          run_vals.append(trial.suggest_int(":".join(keys[ind]), val[0], val[1]))
        elif isinstance(val[0], str):
          run_vals.append(trial.suggest_categorical(":".join(keys[ind]), val))
      else:
        run_vals.append(val)

    # write hyperparameters to file
    output = {}
    for ind in range(len(keys)):
      output = MakeDictionaryEntry(output, keys[ind], run_vals[ind])
    self.tune_architecture_name = f"{self.data_output}/tune_architecture_{self.objective_ind}.yaml"
    MakeDirectories(self.tune_architecture_name)
    with open(self.tune_architecture_name, 'w') as yaml_file:
      yaml.dump(output, yaml_file, default_flow_style=False)

    # Train and performance metrics
    self._TrainAndPerfomanceMetric()

    # Load in metrics
    metrics_name = f"{self.data_output}/metrics_{self.objective_ind}.yaml"
    metric_val =  GetDictionaryEntryFromYaml(metrics_name, self.metric.split(",")[0].split(":"))

    # Increment indices
    self.objective_ind += 1

    return metric_val

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    outputs += [
      f"{self.best_model_output}/{parameters['file_name']}.h5",
      f"{self.best_model_output}/{parameters['file_name']}_architecture.yaml"      
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    inputs = [
      self.parameters,
      self.architecture,
      f"{parameters['file_loc']}/X_train.parquet",
      f"{parameters['file_loc']}/Y_train.parquet", 
      f"{parameters['file_loc']}/wt_train.parquet", 
      f"{parameters['file_loc']}/X_test.parquet",
      f"{parameters['file_loc']}/Y_test.parquet", 
      f"{parameters['file_loc']}/wt_test.parquet",
    ]
    return inputs

  def _SetupTrain(self):
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    t = Train()
    t.Configure(
      {
        "parameters" : self.parameters,
        "architecture" : self.tune_architecture_name,
        "data_output" : self.data_output,
        "use_wandb" : self.use_wandb,
        "disable_tqdm" : self.disable_tqdm,
        "no_plot" : True,
        "save_extra_name" : f"_{self.objective_ind}"
      }
    )
    return t

  def _SetupPerformanceMetrics(self):
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    pf = PerformanceMetrics()
    pf.Configure(
      {
        "model" : f"{self.data_output}/{parameters['file_name']}_{self.objective_ind}.h5",
        "architecture" : self.tune_architecture_name,
        "parameters" : self.parameters,
        "data_output" : self.data_output,
        "save_extra_name" : f"_{self.objective_ind}"
      }
    )
    return pf