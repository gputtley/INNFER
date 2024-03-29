import os

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
    self.job_name = "job.sh"
    self.store_output = True
    self.store_error = True
    self.dry_run = False
    self.sge_queue = "hep.q"
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

  def RunSGE(self):
    """
    Run the batch job using the SGE (Sun Grid Engine) submission system.
    """
    self._CreateBatchJob(self.cmds)

    if self.store_output:
      output_log = self.job_name.replace('.sh','_output.log')
      if os.path.exists(output_log): os.system(f'rm {output_log}')
    else:
      output_log = "/dev/null"
    if self.store_error:
      error_log = self.job_name.replace('.sh','_error.log')
      if os.path.exists(error_log): os.system(f'rm {error_log}')
    else:
      error_log = "/dev/null"

    if not self.dry_run:
      if self.sge_queue != "gpu.q":
        if self.cores>1:
          sub_cmd = f'qsub -e {error_log} -o {output_log} -V -q {self.sge_queue} -pe hep.pe {self.cores} -l h_rt={self.running_hours}:0:0 -l h_vmem={self.memory}G -cwd {self.job_name}'
        else:
          sub_cmd = f'qsub -e {error_log} -o {output_log} -V -q {self.sge_queue} -l h_rt={self.running_hours}:0:0 -l h_vmem={self.memory}G -cwd {self.job_name}'
      else:
        if self.running_hours < 24: self.running_hours = 24
        sub_cmd = f'qsub -e {error_log} -o {output_log} -V -q {self.sge_queue} -l h_rt={self.running_hours}:0:0 -cwd {self.job_name}'
      os.system(sub_cmd)

  def RunCondor(self):
    """
    Run the batch job using the condor submission system.
    """    
    self._CreateBatchJob(self.cmds)

    if self.store_output:
      output_log = self.job_name.replace('.sh','_output.log')
      if os.path.exists(output_log): os.system(f'rm {output_log}')
    else:
      output_log = "/dev/null"
    if self.store_error:
      error_log = self.job_name.replace('.sh','_error.log')
      if os.path.exists(error_log): os.system(f'rm {error_log}')
    else:
      error_log = "/dev/null"
    
    if self.running_hours < 1/3:
      queue = "espresso"
    elif self.running_hours < 1:
      queue = "microcentury"
    elif self.running_hours < 2:
      queue = "longlunch"
    elif self.running_hours < 8:
      queue = "workday"
    elif self.running_hours < 24:
      queue = "tomorrow"
    elif self.running_hours < 72:
      queue = "testmatch"
    else:
      queue = "nextweek"

    sub_file = [
      f"executable = {os.getcwd()}/{self.job_name}",
      f"output                = {os.getcwd()}/{output_log}",
      f"error                 = {os.getcwd()}/{error_log}",
      f"log                   = {os.getcwd()}/{self.job_name.replace('.sh','_condor.log')}",
      #"on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
      #"periodic_release =  (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)",
      f"+JobFlavour = '{queue}'",
      f"RequestCpus = {self.cores}",
      #"Requirements = (OpSysAndVer =?= 'AlmaLinux9')",
      "queue",
    ]

    self._CreateJob(sub_file, self.job_name.replace(".sh",".sub"))

    if not self.dry_run:
      os.system(f"condor_submit {self.job_name.replace('.sh','.sub')}")

  def _CreateJob(self, cmd_list, job_name):
    """
    Create a job script with the specified command list.

    Args:
        cmd_list (list): List of commands to be included in the job script.
    """
    if os.path.exists(job_name): os.system(f'rm {job_name}')
    for cmd in cmd_list:
      os.system(f'echo "{cmd}" >> {job_name}')
    os.system(f'chmod +x {job_name}' % vars())
    print("Created job:",job_name)

  def _CreateBatchJob(self, cmd_list):
    """
    Create a batch job script with additional setup commands.

    Args:
        cmd_list (list): List of commands to be included in the batch job script.
    """
    base_cmds = [
      "#!/bin/bash",
      f"cd {os.getcwd()}",
      "source env.sh",
      "ulimit -s unlimited",
    ]
    self._CreateJob(base_cmds+cmd_list, self.job_name)
