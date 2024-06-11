import copy
import yaml

from useful_functions import StringToFile

class CheckRunAndSubmit():

  def __init__(self, argv, submit = None, specific = "", dry_run = False, points_per_job = 1, job_name = "job"):

    # Add '' when appropriate to command
    self.corrected_argv = []
    for arg in argv:
      if arg.count("=") > 1:
        self.corrected_argv.append(arg.split("=")[0] + "='" + "=".join(arg.split("=")[1:]) + "'")
      else:
        self.corrected_argv.append(arg)
    self.cmd = " ".join(self.corrected_argv)

    # Set up specific dictionary
    if specific == "":
      self.specific_dict = {}
    else:
      self.specific_dict = {i.split("=")[0]:i.split("=")[1].split(",") for i in specific.split(";")}

    # Save variables for later
    self.submit = submit
    self.dry_run = dry_run
    self.points_per_job = points_per_job
    self.job_name = job_name

    # Stores
    self.cmd_store = []
    self.extra_names = []

  def Run(self, loop = {}):

    # If the loop val is not in the specific dictionary do not run
    run = True
    for k, v in self.specific_dict.items():
      if k in loop.keys():
        if str(loop[k]) not in v:
          run = False

    if not run:
      return False

    if run:

      # run if not submit
      if self.submit is None:
        return True

      # set up batch submission command
      to_remove = ["--specific=", "--submit=", "--points-per-job=", "--dry-run"]
      for remove in to_remove:
        remove_str = None
        for string in self.corrected_argv:
          if string.startswith(remove):
            remove_str = copy.deepcopy(string)    
        if remove_str is not None: self.corrected_argv.remove(remove_str)

      specific_sub_string = ";".join([f"{k}={v}" for k, v in loop.items()])
      if specific_sub_string != "":
        specific_sub_string = f" --specific='{specific_sub_string}'"

      # Add command to store
      self.cmd_store.append(f"python3 {' '.join(self.corrected_argv)} {specific_sub_string}")

      # Make extra name
      extra_name = "_".join([f"{k}_{v}" for k, v in loop.items()])
      if extra_name != "":
        extra_name = f"_{extra_name}"
      self.extra_names.append(extra_name)

      # If not enough cmds to do run
      if len(self.cmd_store) < self.points_per_job:
        return False

      # Submit jobs
      self.Sweep()

      return False

  def Sweep(self):

    # If no commands do not run
    if len(self.cmd_store) == 0:
      return None

    # Make combined extra_name
    if len(self.extra_names) == 1:
      extra_name = self.extra_names[0]
    else:
      extra_name = f"{self.extra_names[0]}_to{self.extra_names[-1]}"

    # Open config
    with open(self.submit, 'r') as yaml_file:
      submit_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Submit job
    options = {"submit_to": submit_cfg["name"], "options": submit_cfg["options"],"cmds": self.cmd_store, "job_name": StringToFile(f"{self.job_name}{extra_name}.sh"), "dry_run": self.dry_run}
    from batch import Batch
    sub = Batch(options=options)
    sub.Run()

    # Reset combine store after ran
    self.cmd_store = []
    self.extra_names = []