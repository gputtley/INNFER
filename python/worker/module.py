import importlib
import copy
import yaml
import os

from useful_functions import StringToFile
from batch import Batch

class Module():

  def __init__(
    self,
    argv, 
    args,
    default_args,
    snakemake = None,
    job_name = "job",
    ):

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

  def _MakeCommandOptions(
    self,
    args,
  ):
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

  def _MakeSpecificDictionary(
    self, 
    specific
    ):

    if specific == "":
      specific_dict = {}
    else:
      specific_dict = {i.split("=")[0]:i.split("=")[1].split(",") for i in specific.split(";")}
    return specific_dict

  def _MakeSpecificString(
    self,
    specific,
    ):

    specific_strings = []
    for k, v in specific.items():
      if not isinstance(v, list):
        specific_strings.append(f"{k}={v}")
      else:
        specific_strings.append(f"{k}={','.join(v)}")
    return ";".join(specific_strings)

  def _CheckRunFromSpecific(
    self,
    specific,
    loop,
    ):

    run = True
    for k, v in specific.items():
      if k in loop.keys():
        if str(loop[k]) not in v:
          run = False
    return run

  def _SetupBatchCommand(
    self,
    loop,
    ):

    argv = self._MakeCommandOptions(self.args)
    to_remove = ["--specific=", "--submit=", "--points-per-job=", "--dry-run", "--make-snakemake-inputs", "--snakemake-cfg"]
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

  def _MakeExtraNames(
    self,
    specific,
    ):
    extra_name = "_".join([f"{k}_{v}" for k, v in specific.items()])
    if extra_name != "":
      extra_name = f"_{extra_name}"
    return extra_name


  def Run(
    self,    
    module_name,
    class_name,
    config = {},
    loop = {},
    force = False,
    ):
    
    # If specific does not require the loop to be run than end function
    if not self._CheckRunFromSpecific(self.specific, loop):
      return None

    # If it should be not be submit or be used to make a snakemake file then run
    if self.args.submit is None and (self.snakemake is None or force):

      # Printing loop
      print(f"* Running {self._MakeSpecificString(loop)}")

      # Import modules and initiating class
      module = importlib.import_module(module_name)
      module_class = getattr(module, class_name)
      class_instance = module_class()

      # Configure and run
      class_instance.Configure(config)
      class_instance.Run()
      return None

    # Setup batch command and add it to the command store
    cmd = self._SetupBatchCommand(loop)
    self.cmd_store.append(cmd)

    # Make extra names of cmd
    self.extra_names.append(self._MakeExtraNames(loop))

    # If snakemake gets inputs and outputs
    if self.snakemake is not None:
      module = importlib.import_module(module_name)
      module_class = getattr(module, class_name)
      class_instance = module_class()
      class_instance.Configure(config)
      self.input_store += class_instance.Inputs()
      self.output_store += class_instance.Outputs()
      self.input_store = list(set(self.input_store))
      self.output_store = list(set(self.output_store))

    # Check points per job and whether to add 
    if len(self.cmd_store) < self.args.points_per_job:
      return None

    # Sweep up jobs, either snakemake or submit
    self.Sweep()

  def Sweep(self):

    # If no commands do not run
    if len(self.cmd_store) == 0:
      return None

    # Make combined extra_name
    if len(self.extra_names) == 1:
      extra_name = self.extra_names[0]
    else:
      extra_name = f"{self.extra_names[0]}_to{self.extra_names[-1]}"

    # Direct submission
    if self.args.submit is not None and self.snakemake is None:

      # Open config
      with open(self.args.submit, 'r') as yaml_file:
        submit_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

      # Submit job
      options = {"submit_to": submit_cfg["name"], "options": submit_cfg["options"],"cmds": self.cmd_store, "job_name": StringToFile(f"{self.job_name}{extra_name}.sh"), "dry_run": self.args.dry_run}
      sub = Batch(options=options)
      sub.Run()

    # Add snakemake information
    elif self.snakemake is not None:
      # Make job
      job_name = StringToFile(f"{self.job_name}{extra_name}.sh")
      b = Batch(options={"job_name":job_name})
      b._CreateBatchJob(self.cmd_store)

      # Begin writting rules
      rules = [f"rule {job_name.split('/')[-1].split('.sh')[0]}:"]

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
          for extra_line in submit_cfg["options"]["extra_lines"]:
            rules += [f"    '{extra_line}',"]
      rules[-1] = rules[-1][:-1]

      # Write command
      rules += [
        "  shell:",
        f"    'bash {job_name}'",        
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

