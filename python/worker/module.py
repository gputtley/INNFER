import copy
import importlib
import os
import yaml

from batch import Batch
from useful_functions import StringToFile, CamelToSnake

class Module():

  def __init__(
      self,
      argv, 
      args,
      default_args,
      job_name = "job",
    ):
    """
    A class to facilitate module execution and batch processing.

    Parameters
    ----------
    argv : list
        List of command-line arguments.
    args : argparse.Namespace
        Parsed command-line arguments.
    default_args : argparse.Namespace
        Default command-line arguments.
    job_name : str, optional
        Name of the job (default is "job").
    """

    self.argv = argv
    self.args = args
    self.default_args = default_args

    self.snakemake = args.make_snakemake_inputs
    self.job_name = job_name

    self.specific = self._MakeSpecificDictionary(args.specific)

    self.cmd_store = []
    self.extra_names = []
    self.input_store = []
    self.output_store = []
    self.saved_class = None

  def _CheckRunFromSpecific(self, specific, loop):
    """
    Checks if the current loop should be executed based on specific conditions.

    Parameters
    ----------
    specific : dict
        Dictionary containing specific conditions for job execution.
    loop : dict
        Current loop conditions.

    Returns
    -------
    bool
        True if the loop should be executed, False otherwise.
    """

    run = True
    for k, v in specific.items():
      if k in loop.keys():
        if str(loop[k]) not in v:
          run = False
    return run

  def _MakeCommandOptions(self, args):
    """
    Generates command-line options based on parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    list
        List of command-line options.
    """

    parsers = []
    for attr, value in vars(args).items():
      if value == getattr(self.default_args, attr):
        continue
      if value is None or (value is False and isinstance(value,bool)):
        continue
      attr_name = attr.replace("_","-")
      if value is True and isinstance(value,bool):
        parsers.append(f'--{attr_name}')
      elif isinstance(value, str):
        parsers.append(f'--{attr_name}="{value}"')
      else:
        parsers.append(f'--{attr_name}={value}')
    return parsers

  def _MakeExtraNames(self, specific):
    """
    Generates extra names based on specific conditions.

    Parameters
    ----------
    specific : dict
        Dictionary containing specific conditions for job execution.

    Returns
    -------
    str
        Extra name string generated from specific conditions.
    """

    extra_name = "_".join([f"{k}_{v}" for k, v in specific.items()])
    if extra_name != "":
      extra_name = f"_{extra_name}"
    return extra_name

  def _MakeSpecificDictionary(self, specific):
    """
    Converts specific conditions string into a dictionary format.

    Parameters
    ----------
    specific : str
        String representing specific conditions.

    Returns
    -------
    dict
        Dictionary of specific conditions.
    """

    if specific == "":
      specific_dict = {}
    else:
      specific_dict = {i.split("=")[0]:i.split("=")[1].split(",") for i in specific.split(";")}
    return specific_dict

  def _MakeSpecificString(self, specific):
    """
    Converts specific conditions dictionary into a string format.

    Parameters
    ----------
    specific : dict
        Dictionary containing specific conditions for job execution.

    Returns
    -------
    str
        String representation of specific conditions.
    """

    specific_strings = []
    for k, v in specific.items():
      if not isinstance(v, list):
        specific_strings.append(f"{k}={v}")
      else:
        specific_strings.append(f"{k}={','.join(v)}")
    return ";".join(specific_strings)

  def _SetupBatchCommand(self, loop):
    """
    Sets up the batch command for execution.

    Parameters
    ----------
    loop : dict
        Current loop conditions.

    Returns
    -------
    str
        Batch command string.
    """

    argv = self._MakeCommandOptions(self.args)
    to_remove = ["--specific=", "--submit=", "--points-per-job=", "--dry-run", "--make-snakemake-inputs", "--snakemake-cfg", "--replace-inputs", "--replace-outputs"]
    for remove in to_remove:
      remove_str = None
      for string in argv:
        if string.startswith(remove):
          remove_str = copy.deepcopy(string)
      if remove_str is not None: 
        argv.remove(remove_str)

    specific_sub_string = self._MakeSpecificString(loop)
    if specific_sub_string != "":
      specific_sub_string = f' --specific="{specific_sub_string}"'

    return f"python3 {self.argv[0]} {' '.join(argv)} {specific_sub_string}"


  def _SetupGlobal(self, loop):
    """
    Sets up global variables for the job.

    Parameters
    ----------
    loop : dict
        Current loop conditions.
    """
    # Load global config
    with open("configs/other/global.yaml", 'r') as yaml_file:
      global_config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Set up global variables
    for key, values in global_config.items():
      already_set = False
      for value in values:
        set_var = True
        value_dict = copy.deepcopy(value)
        del value_dict["value"]

        for value_key, value_item in value_dict.items():
          if value_key in loop.keys():
            if value_item != loop[value_key]:
              set_var = False
          else:
            set_var = False

        if set_var and not already_set:
          already_set = True
          os.environ[key] = value["value"]


  def Run(
      self,    
      module_name,
      class_name,
      config = {},
      loop = {},
      force = False,
      save_class = False,
    ):
    """
    Executes the module and optionally submits jobs or creates Snakemake rules.

    Parameters
    ----------
    module_name : str
        Name of the module to import.
    class_name : str
        Name of the class to instantiate.
    config : dict, optional
        Configuration dictionary for the module (default is {}).
    loop : dict, optional
        Loop conditions dictionary (default is {}).
    force : bool, optional
        Force flag to bypass certain conditions (default is False).
    """

    # If specific does not require the loop to be run than end function
    if not self._CheckRunFromSpecific(self.specific, loop):
      return None

    # If it should be not be submit or be used to make a snakemake file then run
    if self.args.submit is None and ((not self.snakemake) or force):

      # Printing loop
      print(f"* Running {self._MakeSpecificString(loop)}")

      # Set up global variables
      self._SetupGlobal(loop)

      # Import modules and initiating class
      if self.saved_class is None:
        module = importlib.import_module(module_name)
        module_class = getattr(module, class_name)
        class_instance = module_class()
      else:
        class_instance = self.saved_class

      # Configure class
      class_instance.Configure(config)

      # Check inputs exist
      for i in class_instance.Inputs():
        if not os.path.isfile(i):
          raise FileNotFoundError(f"The input file '{i}' was not found.")

      # Run
      class_instance.Run()

      # Check outputs exist
      for o in class_instance.Outputs():
        if not os.path.isfile(o):
          raise FileNotFoundError(f"The output file '{o}' was not found.")

      # Save class if required
      if save_class:
        self.saved_class = class_instance
      else:
        self.saved_class = None

      return None

    # Setup batch command and add it to the command store
    cmd = self._SetupBatchCommand(loop)
    self.cmd_store.append(cmd)

    # Make extra names of cmd
    self.extra_names.append(self._MakeExtraNames(loop))

    # If snakemake gets inputs and outputs
    if self.snakemake:
      module = importlib.import_module(module_name)
      module_class = getattr(module, class_name)
      class_instance = module_class()
      class_instance.Configure(config)
      self.input_store += class_instance.Inputs()
      self.output_store += class_instance.Outputs()
      self.input_store = sorted(list(set(self.input_store)))
      self.output_store = sorted(list(set(self.output_store)))
      # if output is in the inputs change to dummy file and add dummy file creation to cmd store
      

    # Check points per job and whether to add 
    if len(self.cmd_store) < self.args.points_per_job:
      return None

    # Sweep up jobs, either snakemake or submit
    self.Sweep()

  def Sweep(self):
    """
    Executes the batch of commands stored in cmd_store.

    If Snakemake is enabled, also generates Snakemake rules based on inputs and outputs.
    """

    # If no commands do not run
    if len(self.cmd_store) == 0:
      return None

    # Make combined extra_name
    if len(self.extra_names) == 1:
      extra_name = self.extra_names[0]
    else:
      extra_name = f"{self.extra_names[0]}_to{self.extra_names[-1]}"

    # Direct submission
    if self.args.submit is not None and (not self.snakemake):

      # Open config
      with open(self.args.submit, 'r') as yaml_file:
        submit_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Make job name
      job_name = StringToFile(f"{self.job_name}{extra_name}")
      if self.args.extra_dir_name != "":
        job_name += f"_{self.args.extra_dir_name}"
      else:
        if self.args.extra_input_dir_name != "":
          job_name += f"_{self.args.extra_input_dir_name}"
        if self.args.extra_output_dir_name != "":
          job_name += f"_{self.args.extra_output_dir_name}"
      if self.args.extra_plot_name != "":
        job_name += f"_{self.args.extra_plot_name}"

      # Submit job
      options = {"submit_to": submit_cfg["name"], "options": submit_cfg["options"],"cmds": self.cmd_store, "job_name": f"{job_name}.sh", "dry_run": self.args.dry_run}
      sub = Batch(options=options)
      sub.Run()

    # Add snakemake information
    elif self.snakemake:

      # Make job
      job_name = StringToFile(f"{self.job_name}{extra_name}")

      rule_name = job_name.split('/')[-1].split('.sh')[0]
      if self.args.extra_dir_name != "":
        rule_name += f"_{self.args.extra_dir_name}"
        job_name += f"_{self.args.extra_dir_name}"
      else:
        if self.args.extra_input_dir_name != "":
          rule_name += f"_{self.args.extra_input_dir_name}"
          job_name += f"_{self.args.extra_input_dir_name}"
        if self.args.extra_output_dir_name != "":
          rule_name += f"_{self.args.extra_output_dir_name}"
          job_name += f"_{self.args.extra_output_dir_name}"
      if self.args.extra_plot_name != "":
        rule_name += f"_{self.args.extra_plot_name}"
        job_name += f"_{self.args.extra_plot_name}"

      job_name += ".sh"
      b = Batch(options={"job_name":job_name})
      b._CreateBatchJob(self.cmd_store)

      # Begin writting rules
      rules = [f"rule {rule_name}:"]

      # Write inputs
      rules += ["  input:"]
      for inputs in self.input_store:
        rules += [f"    '{inputs}',"]
      rules[-1] = rules[-1][:-1]

      # Write outputs
      rules += ["  output:"]
      for outputs in self.output_store:
        rules += [f"    '{outputs}',"]
      rules[-1] = rules[-1][:-1]

      # Write batch settings
      with open(self.args.submit, 'r') as yaml_file:
        submit_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
      if submit_cfg["name"] != "condor":
        raise ValueError("Currently only condor snakemake submission supported")
      if "options" in submit_cfg.keys():
        if "extra_lines" in submit_cfg["options"].keys():
          rules += ["  params:"]
          rules += ["    submit_options=["]
          for extra_line in submit_cfg["options"]["extra_lines"]:
            key = extra_line.split("=")[0].replace(" ","")
            if not key.startswith("+"):
              key = CamelToSnake(key)
            val = extra_line.split("=")[1].replace(" ","")
            rules += [f"      '{key}={val}',"]
      rules[-1] = rules[-1][:-1]
      rules += ["    ]"]

      error_name = job_name.replace('.sh','_error.log')
      output_name = job_name.replace('.sh','_output.log')
      # Write command
      rules += [
        "  shell:",
        f"    'bash {job_name}  > {output_name} 2> {error_name}'",        
      ]

      # Write rules
      snakemake_file = f'{"/".join(self.job_name.split("/")[:-1])}/innfer_SnakeMake.txt'
      b_rules = Batch()
      if os.path.isfile(snakemake_file):
        rules = [""] + rules
      b_rules._CreateJob(rules, snakemake_file, delete_job=False)   

    # Reset combine store after ran
    self.cmd_store = []
    self.extra_names = []
    self.input_store = []
    self.output_store = []

