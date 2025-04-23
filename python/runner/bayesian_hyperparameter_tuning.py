import optuna
import os
import wandb
import yaml

from random_word import RandomWords

from train_density import TrainDensity
from density_performance_metrics import DensityPerformanceMetrics
from useful_functions import FindKeysAndValuesInDictionaries, MakeDictionaryEntry, GetDictionaryEntryFromYaml, MakeDirectories

class BayesianHyperparameterTuning():

  def __init__(self):
    """
    A template class.
    """
    # Required input which is the location of a file
    self.cfg = None
    self.parameters = None    
    self.tune_architecture = None

    # other
    self.best_model_output = None
    self.metric = ""

    self.file_name = None
    self.data_input = "data/"
    self.use_wandb = False
    self.wandb_project_name = "innfer"
    self.wandb_submit_name = "innfer"
    self.verbose = True
    self.disable_tqdm = False
    self.data_output = "data/"
    self.density_performance_metrics = None
    self.n_trials = 10

    self.objective_ind = 0
    self.tune_architecture_name = None


  def _TrainAndPerformanceMetric(self):
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
    model_output_name = f"{self.best_model_output}/{self.file_name}.h5"
    model_architecture_name = f"{self.best_model_output}/{self.file_name}_architecture.yaml"
    best_model_name = f"{self.data_output}/{self.file_name}_{best_trial}.h5"
    best_architecture_name = f"{self.data_output}/{self.file_name}_{best_trial}_architecture.yaml"

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
    self._TrainAndPerformanceMetric()

    # Load in metrics
    metrics_name = f"{self.data_output}/metrics_{self.objective_ind}.yaml"
    metric_val =  GetDictionaryEntryFromYaml(metrics_name, self.metric.split(",")[0].split(":"))
    if metric_val is None:
      print("- Metric not found, converting to loss_test")
      self.metric = "loss_test,min"
      metric_val = GetDictionaryEntryFromYaml(metrics_name, self.metric.split(",")[0].split(":"))

    # Increment indices
    self.objective_ind += 1

    return metric_val


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    outputs += [
      f"{self.best_model_output}/{self.file_name}.h5",
      f"{self.best_model_output}/{self.file_name}_architecture.yaml"      
    ]
    return outputs


  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = []

    # Add config
    inputs += [self.cfg]

    # Add parameters
    inputs += [self.parameters]

    # Add architecture
    inputs += [self.tune_architecture]

    # Add other inputs
    t = self._SetupTrain()
    pf = self._SetupPerformanceMetrics()
    t_inputs = t.Inputs()
    t_inputs = [i for i in t_inputs if i is not None]
    t_outputs = t.Outputs()
    pf_inputs = pf.Inputs()
    pf_unique = [i for i in pf_inputs if i not in t_outputs and i is not None]
    inputs = list(set(inputs + t_inputs + pf_unique))

    return inputs


  def _SetupTrain(self):

    t = TrainDensity()
    t.Configure(     
      {     
        "parameters" : self.parameters,
        "architecture" : self.tune_architecture_name,
        "file_name" : self.file_name,
        "data_input" : f"{self.data_input}/{self.file_name}/density",
        "data_output" : self.data_output,
        "no_plot" : True,
        "disable_tqdm" : self.disable_tqdm,
        "use_wandb" : self.use_wandb,
        "initiate_wandb" : self.use_wandb,
        "wandb_project_name" : self.wandb_project_name,
        "wandb_submit_name" : self.wandb_submit_name,
        "save_extra_name" : f"_{self.objective_ind}",
        "verbose" : self.verbose,        
      }
    )

    return t


  def _SetupPerformanceMetrics(self):

    pf = DensityPerformanceMetrics()
    pf.Configure(
      {
        "cfg" : self.cfg,
        "file_name" : self.file_name,
        "parameters" : self.parameters,
        "model_input" : self.data_output,
        "data_input" : self.data_input,
        "data_output" : self.data_output,
        "do_inference": "inference" in self.density_performance_metrics,
        "do_loss": "loss" in self.density_performance_metrics,
        "do_histogram_metrics": "histogram" in self.density_performance_metrics,
        "do_multidimensional_dataset_metrics": "multidim" in self.density_performance_metrics,
        "save_extra_name" : f"_{self.objective_ind}",
        "verbose" : self.verbose,     
      }
    )

    return pf