import argparse
import yaml
import glob
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('-c','--cfg', help= 'Config for running',  default=None)
parser.add_argument('--submit', help= 'Batch to submit to,',  default="SGE")
parser.add_argument('--options', help= 'Options to be parsed to innfer without the cfg, submit, step and substep options',  default="")
parser.add_argument('--first-step', help= 'Step to begin with.',  default="PreProcess")
parser.add_argument('--last-step', help= 'Step to end with.',  default="ValidateInference")
parser.add_argument('--first-sub-step', help= 'Step to begin with.',  default="InitialFit")
parser.add_argument('--last-sub-step', help= 'Step to end with.',  default="Plot")
parser.add_argument('--add-train-option', help= 'Additional option for the train step.',  default="--sge-queue=gpu.q")
args = parser.parse_args()

total_mins_per_job_till_end = 300
mins_till_run = 1

with open(args.cfg, 'r') as yaml_file:
  cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

steps = {
  "PreProcess" : [None],
  "Train" : [None],
  "ValidateGeneration" : [None],
  "ValidateInference" : [
    "InitialFit",
    "Scan",
    "Collect",
    "Plot"
  ]
}

# trim steps and sub_steps
trimmed_steps = {}
for step_ind, (step, sub_steps) in enumerate(steps.items()):
  if step_ind >= list(steps.keys()).index(args.first_step) and step_ind <= list(steps.keys()).index(args.last_step):
    trimmed_steps[step] = []
  else:
    continue
  for sub_step_ind, sub_step in enumerate(sub_steps):
    if sub_step == None:
      trimmed_steps[step].append(None)
      continue
    if sub_step_ind >= sub_steps.index(args.first_sub_step) and sub_step_ind <= sub_steps.index(args.last_sub_step):
      trimmed_steps[step].append(sub_step)



for step, sub_steps in trimmed_steps.items():
  for sub_step in sub_steps:

    print(f"- Running step: {step}")

    if sub_step == None:
      sub_step_option = ""
      job_files = f"jobs/{cfg['name']}/{step.lower()}/innfer_{step.lower()}_{cfg['name']}_*.sh"
    else:
      print(f"- Running sub-step: {sub_step}")
      sub_step_option = f"--sub-step={sub_step}"
      job_files = f"jobs/{cfg['name']}/{step.lower()}/{sub_step.lower()}/innfer_{step.lower()}_{sub_step.lower()}_{cfg['name']}_*.sh"

    # Need to clear log files first
    for file in glob.glob(f"{job_files}"):
      os.system(f"rm {file}")
      os.system(f"rm {file.replace('.sh','_output.log')}")
      os.system(f"rm {file.replace('.sh','_error.log')}")

    if step == "Train":
      args.options += args.add_train_option

    cmd = f"python3 scripts/innfer.py {args.options} --cfg={args.cfg} --step={step} {sub_step_option} --submit={args.submit}"
    print(cmd)
    os.system(cmd)
    time.sleep(mins_till_run * 60)

    start_time = time.time()
    while (time.time()-start_time) < total_mins_per_job_till_end*60:

      print(f"Monitoring time: {time.time()-start_time} secs")
      all_jobs_finished = True
      for file in glob.glob(f"{job_files}"):
        log_file = file.replace(".sh","_output.log")
        print(log_file)

        if not os.path.exists(log_file):
          all_jobs_finished = False
          time.sleep(mins_till_run * 60)
          continue          

        finished = False
        with open(log_file, 'r') as file:
          for line_number, line in enumerate(file, start=1):
            if line.strip() == "- Finished running without error":
              finished = True

        if not finished:
          all_jobs_finished = False
          time.sleep(mins_till_run * 60)
          continue

      if all_jobs_finished: break
