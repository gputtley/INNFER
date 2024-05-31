import yaml

from useful_functions import MakeDirectories

class PerformanceMetrics():

  def __init__(self):
    """
    A template class.
    """
    # Default values - these will be set by the configure function
    self.model = None
    self.parameters = None
    self.architecture = None

    self.data_output = "data/"
    self.verbose = True
    self.do_loss = True
    self.do_chi_squared = True
    self.do_bdt_separation = True
    self.save_extra_name = ""

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

    # Load the architecture in
    if self.verbose:
      print("- Loading in the architecture")
    with open(self.architecture, 'r') as yaml_file:
      architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Build model
    if self.verbose:
      print("- Building the model")
    from network import Network
    network = Network(
      f"{parameters['file_loc']}/X_train.parquet",
      f"{parameters['file_loc']}/Y_train.parquet", 
      f"{parameters['file_loc']}/wt_train.parquet", 
      f"{parameters['file_loc']}/X_test.parquet",
      f"{parameters['file_loc']}/Y_test.parquet", 
      f"{parameters['file_loc']}/wt_test.parquet",
      options = {
        **architecture,
        **{
          "data_parameters" : parameters
        }
      }
    )  
    
    # Loading model
    if self.verbose:
      print("- Loading the model")
    network.Load(name=self.model)

    # Set up metrics dictionary
    metrics = {} 

    # Get the loss values
    if self.do_loss:
      if self.verbose:
        print("- Getting the losses")
      metrics["loss_train"] = network.GetLoss(dataset="train")
      metrics["loss_test"] = network.GetLoss(dataset="test")

    # Get BDT separation metric
    if self.do_bdt_separation:
      if self.verbose:
        print("- Getting BDT separation")
      metrics["bdt_separation_train"] = network.GetAUC(dataset="train")
      metrics["bdt_separation_test"] = network.GetAUC(dataset="test")


    # Get histogram metrics to run
    histogram_metrics = []
    if self.do_chi_squared:
      histogram_metrics.append("chi_squared")

    if len(histogram_metrics):
      if self.verbose:
        print("- Getting histogram metrics")
      hist_metrics = network.GetHistogramMetric(metric=histogram_metrics)

      # Add hist_metric sums
      summed_hist_metrics = {}
      for metric_name, metric_results in hist_metrics.items():
        summed_hist_metrics[metric_name] = {"total":0.0}
        for k1, v1 in metric_results.items():
          for k2, v2 in v1.items():
            summed_hist_metrics[metric_name]["total"] += v2
            k1_sum_name = f"{k1}_sum"
            k2_sum_name = f"{k2}_sum"
            if k1_sum_name not in summed_hist_metrics[metric_name].keys():
              summed_hist_metrics[metric_name][k1_sum_name] = v2*1.0
            else:
              summed_hist_metrics[metric_name][k1_sum_name] += v2
            if k2_sum_name not in summed_hist_metrics[metric_name].keys():
              summed_hist_metrics[metric_name][k2_sum_name] = v2*1.0
            else:
              summed_hist_metrics[metric_name][k2_sum_name] += v2

      metrics = {**metrics, **hist_metrics}

    # Write to yaml
    if self.verbose:
      print("- Writing metrics yaml")
    output_name = f"{self.data_output}/metrics{self.save_extra_name}.yaml"
    MakeDirectories(output_name)
    with open(output_name, 'w') as yaml_file:
      yaml.dump(metrics, yaml_file, default_flow_style=False) 

    # Print metrics
    for metric in sorted(list(metrics.keys())):
      if not isinstance(metrics[metric], dict):
        print(f"{metric} : {metrics[metric]}")
      else:
        print(f"{metric} :")
        for k1 in sorted(list(metrics[metric].keys())):
          if not isinstance(metrics[metric][k1], dict):
            print(f"  {k1} : {metrics[metric][k1]}")
          else:
            print(f"  {k1} :")
            for k2 in sorted(list(metrics[metric][k1].keys())):
              print(f"    {k2} : {metrics[metric][k1][k2]}")          


  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = [
      f"{self.data_output}/metrics{self.save_extra_name}.yaml"
    ]
    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    inputs = [
      self.model,
      self.architecture,
      self.parameters,
      f"{parameters['file_loc']}/X_train.parquet",
      f"{parameters['file_loc']}/Y_train.parquet", 
      f"{parameters['file_loc']}/wt_train.parquet", 
      f"{parameters['file_loc']}/X_test.parquet",
      f"{parameters['file_loc']}/Y_test.parquet", 
      f"{parameters['file_loc']}/wt_test.parquet",
    ]
    return inputs

        