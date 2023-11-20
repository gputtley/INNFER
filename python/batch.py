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
    if self.submit_to == "SGE":
      self.RunSGE()

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
      if self.cores>1:
        sub_cmd = f'qsub -e {error_log} -o {output_log} -V -q {self.sge_queue} -pe hep.pe {self.cores} -l h_rt={self.running_hours}:0:0 -l h_vmem={self.memory}G -cwd {self.job_name}'
      else:
        sub_cmd = f'qsub -e {error_log} -o {output_log} -V -q {self.sge_queue} -l h_rt={self.running_hours}:0:0 -l h_vmem={self.memory}G -cwd {self.job_name}'
      os.system(sub_cmd)

  def _CreateJob(self, cmd_list):
    """
    Create a job script with the specified command list.

    Args:
        cmd_list (list): List of commands to be included in the job script.
    """
    if os.path.exists(self.job_name): os.system(f'rm {self.job_name}')
    for cmd in cmd_list:
      os.system(f'echo "{cmd}" >> {self.job_name}')
    os.system(f'chmod +x {self.job_name}' % vars())
    print("Created job:",self.job_name)

  def _CreateBatchJob(self, cmd_list):
    """
    Create a batch job script with additional setup commands.

    Args:
        cmd_list (list): List of commands to be included in the batch job script.
    """
    base_cmds = [
      "#!/bin/bash",
      "source env.sh",
      "ulimit -s unlimited",
    ]
    self._CreateJob(base_cmds+cmd_list)
