import os

from useful_functions import MakeDirectories

class Batch():
  """
  Batch class for creating and running batch jobs on a cluster.
  """
  def __init__(self, options={}):
    """
    Initialize the Batch class.

    Args:
        options (dict): Dictionary of options for the batch job.
    """
    self.cmds = []
    self.submit_to = "SGE"
    self.running_hours = 3
    self.memory = 24
    self.cores = 1
    self.options = {}

    self.job_name = "job.sh"
    self.dry_run = False
    self.use_environment_container = False
    self._SetOptions(options)

  def _SetOptions(self, options):
    """
    Set options for the validation process.

    Args:
        options (dict): Dictionary of options for the validation process.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):
    """
    Run the batch job based on the specified submission system.
    """
    split_directory = self.job_name.split("/")
    for ind in range(len(split_directory)-1):
      directory = "/".join(split_directory[:ind+1])
      if not os.path.isdir(directory):
        os.system(f"mkdir {directory}")

    if self.submit_to == "SGE":
      self.RunSGE()
    if self.submit_to == "condor":
      self.RunCondor()
    if self.submit_to == "cern_condor":
      self.RunCondor(cern=True)


  def RunSGE(self):
    """
    Run the batch job using the SGE (Sun Grid Engine) submission system.
    """
    self._CreateBatchJob(self.cmds)

    output_log = self.job_name.replace('.sh','_output.log')
    if os.path.exists(output_log): os.system(f'rm {output_log}')
    error_log = self.job_name.replace('.sh','_error.log')
    if os.path.exists(error_log): os.system(f'rm {error_log}')

    if not self.dry_run:
      if self.cores>1:
        sub_cmd = f'qsub -e {error_log} -o {output_log} -V -q {self.options["sge_queue"]} -pe hep.pe {self.options["cores"]} -l h_rt={self.options["running_hours"]}:0:0 -l h_vmem={self.options["self.memory"]}G -cwd {self.job_name}'
      else:
        sub_cmd = f'qsub -e {error_log} -o {output_log} -V -q {self.options["sge_queue"]} -l h_rt={self.options["running_hours"]}:0:0 -l h_vmem={self.options["self.memory"]}G -cwd {self.job_name}'
      os.system(sub_cmd)


  def RunCondor(self, cern=False):
    """
    Run the batch job using the condor submission system.
    """    
    self._CreateBatchJob(self.cmds)

    output_log = self.job_name.replace('.sh','_output.log')
    if os.path.exists(output_log): os.system(f'rm {output_log}')
    error_log = self.job_name.replace('.sh','_error.log')
    if os.path.exists(error_log): os.system(f'rm {error_log}')
    
    exc_and_out = [
      f"executable = {os.getcwd()}/{self.job_name}",
      f"output     = {os.getcwd()}/{output_log}",
      f"error      = {os.getcwd()}/{error_log}",
      f"log        = {os.getcwd()}/{self.job_name.replace('.sh','_condor.log')}",
    ]

    if self.use_environment_container:
      exc_and_out += [
        "transfer_input_files     = innfer_env_container.tar.gz",
        "should_transfer_files    = YES",
        "when_to_transfer_output  = ON_EXIT",
      ]

    if "extra_lines" in self.options.keys():
      extra_lines = self.options["extra_lines"]

    sub_options = []
    if cern and "running_hours" in self.options.keys():
      if self.options["running_hours"] < 1/3:
        queue = "espresso"
      elif self.options["running_hours"] < 1:
        queue = "microcentury"
      elif self.options["running_hours"] < 2:
        queue = "longlunch"
      elif self.options["running_hours"] < 8:
        queue = "workday"
      elif self.options["running_hours"] < 24:
        queue = "tomorrow"
      elif self.options["running_hours"] < 72:
        queue = "testmatch"
      else:
        queue = "nextweek"

      sub_options = [
        f"+JobFlavour = '{queue}'",
      ]

    sub_file = exc_and_out + extra_lines + sub_options + ["queue"]

    self._CreateJob(sub_file, self.job_name.replace(".sh",".sub"))

    if not self.dry_run:
      os.system(f"condor_submit {self.job_name.replace('.sh','.sub')}")


  def _CreateJob(self, cmd_list, job_name, delete_job=True):
    """
    Create a job script with the specified command list.

    Args:
        cmd_list (list): List of commands to be included in the job script.
    """
    MakeDirectories(job_name)
    if os.path.exists(job_name) and delete_job: os.system(f'rm {job_name}')
    for cmd in cmd_list:
      prep_cmd = cmd.replace('"','\\"')
      os.system(f'echo "{prep_cmd}" >> {job_name}')
    os.system(f'chmod +x {job_name}' % vars())
    if delete_job:
      print("Created job:",job_name)
    else:
      print("Adding to:",job_name)


  def _CreateBatchJob(self, cmd_list):
    """
    Create a batch job script with additional setup commands.

    Args:
        cmd_list (list): List of commands to be included in the batch job script.
    """

    #data_dir = os.getenv('DATA_DIR')
    prep_data_dir = os.getenv('PREP_DATA_DIR')
    eval_data_dir = os.getenv('EVAL_DATA_DIR')
    plots_dir = os.getenv('PLOTS_DIR')
    models_dir = os.getenv('MODELS_DIR')

    base_cmds = [
      "#!/bin/bash",
    ]

    if self.use_environment_container:
      base_cmds += [
        "echo 'Setting up environment from container...'",
        "mkdir -p innfer_env_container",
        "tar -xf innfer_env_container.tar.gz -C innfer_env_container",
        "innfer_env_container/bin/conda-unpack",
        "source innfer_env_container/bin/activate",
      ]

    base_cmds += [
      f"cd {os.getcwd()}",
    ]

    if not self.use_environment_container:
      base_cmds += [
        "source env.sh",
      ]
    else:
      base_cmds += [
        "source env.sh no_conda",
      ]

    base_cmds += [
      "ulimit -s unlimited",
      f"export PREP_DATA_DIR={prep_data_dir}",
      f"export EVAL_DATA_DIR={eval_data_dir}",
      f"export PLOTS_DIR={plots_dir}",
      f"export MODELS_DIR={models_dir}",
    ]
    self._CreateJob(base_cmds+cmd_list, self.job_name)
