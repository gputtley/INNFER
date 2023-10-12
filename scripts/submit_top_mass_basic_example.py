import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--step', help= 'step to run', type=str, default="train", choices=["train","infer"])
parser.add_argument('--options', help= 'extra options to run', type=str, default="")
parser.add_argument('--clear', help= 'Clear jobs',  action='store_true')
parser.add_argument('--dry-run', help= 'Do not run jobs',  action='store_true')
args = parser.parse_args()

if not os.path.isdir("jobs"):
  os.system("mkdir jobs")
elif args.clear:
  os.system("rm jobs/*")

def CreateJob(name,cmd_list):
  if os.path.exists(name): os.system('rm %(name)s' % vars())
  for cmd in cmd_list:
    os.system('echo "%(cmd)s" >> %(name)s' % vars())
  os.system('chmod +x %(name)s' % vars())
  print("Created job:",name)

def CreateBatchJob(name,cmd_list):
  base_cmds = [
    "#!/bin/bash",
    "conda activate",
    "source setup.sh",
    "ulimit -s unlimited",
  ]
  CreateJob(name,base_cmds+cmd_list)

def SubmitBatchJob(name,time=180,memory=24,cores=1):
  #error_log = name.replace('.sh','_error.log')
  output_log = name.replace('.sh','_output.log')
  #if os.path.exists(error_log): os.system('rm %(error_log)s' % vars())
  error_log="/dev/null"
  if os.path.exists(output_log): os.system('rm %(output_log)s' % vars())
  if cores>1: os.system('qsub -e %(error_log)s -o %(output_log)s -V -q hep.q -pe hep.pe %(cores)s -l h_rt=0:%(time)s:0 -l h_vmem=%(memory)sG -cwd %(name)s' % vars())
  else: os.system('qsub -e %(error_log)s -o %(output_log)s -V -q hep.q -l h_rt=0:%(time)s:0 -l h_vmem=%(memory)sG -cwd %(name)s' % vars())

file_name = "scripts/top_mass_basic_example_clean.py"
job_name = "top_mass_basic_example"

# find indices
inds = [int(f.split(job_name+"_")[1].split(".sh")[0]) for f in os.listdir("jobs") if f.startswith(job_name+"_") and ".sh" in f]
inds += [0]
ind = max(inds) + 1

if args.step == "train":
  cmds = [
    f"python3 {file_name} {args.options} --skip-closure --skip-generation --skip-probability --skip-inference",
  ]
  jn = "jobs/" + job_name + "_" + str(ind) + ".sh"
  CreateBatchJob(jn, cmds)
  if not args.dry_run: SubmitBatchJob(jn)
  ind += 1

elif args.step == "infer":

  true_mass_plot = [171.0,172.0,173.0,174.0]
  sig_frac_plot = [0.1,0.2,0.3]

  if not "--use-signal-fraction" in args.options:
    sig_frac_plot = [None]

  add_options = ""
  for tm in true_mass_plot:
    for sf in sig_frac_plot:
      if sf == None:
        add_options = f"--only-infer-true-mass={tm}"
        q = 180
      else:
        add_options = f"--only-infer-true-mass={tm} --only-infer-signal-fraction={sf}"
        q = 600

      cmds = [
        f"python3 {file_name} --skip-initial-distribution --load-model {args.options} {add_options}"
      ]
      jn = "jobs/" + job_name + "_" + str(ind) + ".sh"
      CreateBatchJob(jn, cmds)
      if not args.dry_run: SubmitBatchJob(jn, time=q)      
      ind += 1