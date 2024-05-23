import copy
import os

def CheckRunAndSubmit(
  argv, 
  loop = {}, 
  specific = "",
  job_name = "job",
  dry_run = True
  ):

  # Add '' when appropriate to command
  corrected_argv = []
  for arg in argv:
    if arg.count("=") > 1:
      corrected_argv.append(arg.split("=")[0] + "='" + "=".join(arg.split("=")[1:]) + "'")
    else:
      corrected_argv.append(arg)
  cmd = " ".join(corrected_argv)

  # Set up specific dictionary
  if specific == "":
    specific_dict = {}
  else:
    specific_dict = {i.split("=")[0]:i.split("=")[1].split(",") for i in specific.split(";")}

  # If the loop val is not in the specific dictionary do not run
  run = True
  for k, v in specific_dict.items():
    if k in loop.keys():
      if str(loop[k]) not in v:
        run = False
  if run == False:
    return False

  else:

    # run if not submit
    if "--submit" not in cmd:
      return True

    # set up batch submission command
    specific_str = None
    sub_str = None
    for string in corrected_argv:
      if string.startswith("--specific="):
        specific_str = copy.deepcopy(string)
      if string.startswith("--submit="):
        sub_str = copy.deepcopy(string)
        sub_type = string.split("--submit=")[1]
    if specific_str is not None: corrected_argv.remove(specific_str)
    if sub_str is not None: corrected_argv.remove(sub_str)
    specific_sub_string = ";".join([f"{k}={v}" for k, v in loop.items()])
    if specific_sub_string != "":
      specific_sub_string = f" --specific='{specific_sub_string}'"
    sub_cmd = f"python3 {' '.join(corrected_argv)} {specific_sub_string}"  

    # Make extra name
    extra_name = "_".join([f"{k}_{v}" for k, v in loop.items()])
    if extra_name != "":
      extra_name = f"_{extra_name}"

    # Submit job
    options = {"submit_to": sub_type, "cmds": [sub_cmd], "job_name": StringToFile(f"{job_name}{extra_name}.sh"), "dry_run": dry_run}
    from batch import Batch
    sub = Batch(options=options)
    sub.Run()

    return False

def StringToFile(string):
  string = string.replace(",","_").replace(";","_").replace(">=","_geq_").replace("<=","_leq_").replace(">","_g_").replace("<","_l_").replace("!=","_noteq_").replace("=","_eq_")
  if string.count(".") > 1:
    string = f'{"p".join(string.split(".")[:-1])}.{string.split(".")[-1]}'
  return string

def MakeDirectories(file_loc):
  """
  Make directories.

  Args:
      file_loc (str): File location.

  Returns:
      None
  """
  if file_loc[0] == "/":
    initial = "/"
    file_loc = file_loc[1:]
  else:
    initial = ""

  splitting = file_loc.split("/")
  for ind in range(len(splitting)):

    # Skip if file
    if "." in splitting[ind]: continue

    # Get full directory
    full_dir = initial + "/".join(splitting[:ind+1])

    # Make directory if it is missing
    if not os.path.isdir(full_dir): 
      os.system(f"mkdir {full_dir}")