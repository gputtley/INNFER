import os

class Batch():
  
  def __init__(self):

    self.cmds = []
    self.submit_to = "SGE"
    self.running_hours = 3
    self.memory = 24
    self.cores = 1
    self.job_name = "job.sh"
    self.store_output = True
    self.store_error = False
    self.dry_run = False

  def Run(self):
    if self.submit_to == "SGE":
      self.RunSGE()

  def RunSGE(self):
    self._CreateBatchJob(self.cmds)
    if self.store_output:
      output_log = self.job_name.replace('.sh','_output.log')
    else:
      output_log = "/dev/null"
    if self.store_error:
      error_log = self.job_name.replace('.sh','_error.log')
    else:
      error_log = "/dev/null"
    if not self.dry_run:
      if self.cores>1: 
        os.system(f'qsub -e {error_log} -o {output_log} -V -q hep.q -pe hep.pe {self.cores} -l h_rt=0:{self.running_hours}:0 -l h_vmem={self.memory}G -cwd {self.job_name}')
      else: 
        os.system(f'qsub -e {error_log} -o {output_log} -V -q hep.q -l h_rt=0:{self.running_hours}:0 -l h_vmem={self.memory}G -cwd {self.job_name}')

  def _CreateJob(self, cmd_list):
    if os.path.exists(self.job_name): os.system(f'rm {self.job_name}')
    for cmd in cmd_list:
      os.system(f'echo "{cmd}" >> {self.job_name}')
    os.system(f'chmod +x {self.job_name}' % vars())
    print("Created job:",self.job_name)

  def _CreateBatchJob(self, cmd_list):
    base_cmds = [
      "#!/bin/bash",
      "conda activate",
      "source env.sh",
      "ulimit -s unlimited",
    ]
    self._CreateJob(base_cmds+cmd_list)
