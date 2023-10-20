"""
Title: Top mass basic example:
Author: George Uttley

This is a simple example where we try and infer the mass of the top quark 
from one reconstructed mass like context variable using the bayesflow package.

The code can perform 3 main running schemes:
1)  Inferring the top mass from just the Gaussian signal
2)  Inferring the top mass from a Gaussian signal on top of an exponentially 
    falling background.
3)  Inferring the top mass and the signal fraction from a Gaussian signal on top 
    of an exponentially falling background.

There are also sub options for running using discrete input values and for
changing the input signal fractions, in the case of 2.
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--use-discrete-true-mass', help= 'Use discrete true mass',  action='store_true')
parser.add_argument('--use-signal-fraction', help= 'Use the signal fraction as a context parameters, otherwise set to signal_fraction',  action='store_true')
parser.add_argument('--use-discrete-signal-fraction', help= 'Use discrete signal fractions',  action='store_true')
parser.add_argument('--add-background', help= 'Add a falling background',  action='store_true')
parser.add_argument('--load-model', help= 'Load model from file',  action='store_true')
parser.add_argument('--skip-initial-distribution', help= 'Skip making the initial distribution plots',  action='store_true')
parser.add_argument('--skip-closure', help= 'Skip making the closure plots',  action='store_true')
parser.add_argument('--skip-generation', help= 'Skip making the generation plots',  action='store_true')
parser.add_argument('--skip-probability', help= 'Skip draw the probability',  action='store_true')
parser.add_argument('--skip-inference', help= 'Skip performing the inference',  action='store_true')
parser.add_argument('--skip-comparison', help= 'Skip making the comparison plot, if running inference',  action='store_true')
parser.add_argument('--signal-fraction', help= 'Signal fraction value to take if fixed', type=float, default=0.3)
parser.add_argument('--only-infer-true-mass', help= 'Only infer this value of the true mass', type=float, default=None)
parser.add_argument('--only-infer-signal-fraction', help= 'Only infer this value of the signal fraction', type=float, default=None)
parser.add_argument('--add-true-pdf', help= 'Add true pdf in the most simplest case to prob and lkld plot',  action='store_true')
parser.add_argument('--plot-true-masses', help= 'True masses to plot', type=str, default="171.0,172.0,173.0,174.0")
parser.add_argument('--plot-signal-fractions', help= 'True masses to plot', type=str, default="0.1,0.2,0.3")
parser.add_argument('--submit', help= 'Batch to submit to', type=str, default=None)
args = parser.parse_args()

### Set running information ###
if args.use_signal_fraction:
  args.add_background = True
if args.add_true_pdf and args.add_background:
  print("WARNING: Adding the true pdf is not compatible with adding the background")
  args.add_true_pdf = False

### Make name ###
name = "top_mass_basic_example"
if args.use_signal_fraction:
  name += "_with_sig_frac"
elif args.add_background:
  name += "_with_bkg"
if args.use_discrete_signal_fraction and args.use_discrete_true_mass:
  name += "_discrete_sig_frac_and_true_mass"
elif args.use_discrete_signal_fraction:
  name += "_discrete_sig_frac"
elif args.use_discrete_true_mass:
  name += "_discrete_true_mass"
if not os.path.isdir("plots/"+name):
  os.system("mkdir plots/"+name)

### Submit to batch ###
if args.submit != None:
  import sys
  from batch import Batch
  argsparse_list = [i for i in sys.argv if "--submit" not in i]
  sub = Batch()
  sub.cmds = ["python3 " + " ".join(argsparse_list)]
  sub.submit_to = args.submit
  inds = [int(f.split(name+"_")[1].split(".sh")[0]) for f in os.listdir("jobs") if f.startswith(name+"_") and ".sh" in f]
  inds += [0]
  ind = max(inds) + 1
  sub.job_name = "jobs/" + name + "_" + str(ind) + ".sh"
  sub.dry_run = False
  sub.Run()
  exit()

print("- Importing packages")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import bayesflow as bf
import numpy as np
import mplhep as hep
import tensorflow as tf
from data_processor import DataProcessor
from model import Model
from inference import Inference

tf.random.set_seed(42) 
tf.keras.utils.set_random_seed(42)
hep.style.use("CMS")

### Data generation ###
print("- Making datasets")
class MakeDatasets():
  def __init__(self):
    self.array_size = 10**6
    self.return_sig_frac = False
    self.true_mass_ranges = [165.0,180.0]
    self.sig_frac_ranges = [0.0,0.5]
    self.discrete_true_mass = False
    self.discrete_sig_frac = False
    self.discrete_true_masses = [
      165.5,166.5,167.5,168.5,169.5,
      170.5,171.5,172.5,173.5,174.5,
      175.5,176.5,177.5,178.5,179.5,
    ]
    self.discrete_sig_fracs = [
      0.05,0.15,0.25,0.35,0.45
    ]
    self.fix_bkg_events = False # for discrete only
    self.sig_res = 0.01
    self.bkg_lambda = 0.1
    self.bkg_const = 160.0
    self.bkg_ranges = [160.0,185.0]
    self.random_seed = 42
    
  def SetParameters(self,config_dict):
    for key, value in config_dict.items():
      setattr(self, key, value)

  def RandomSignalPlusBackgroundEvent(self, mass, sig_frac):
    if np.random.rand() < sig_frac:
      x = np.random.normal(mass, self.sig_res*mass)
    else:
      x = np.random.exponential(scale=1/self.bkg_lambda) + self.bkg_const
      while x < self.bkg_ranges[0] or x > self.bkg_ranges[1]:
        x = np.random.exponential(scale=1/self.bkg_lambda) + self.bkg_const 
    return x

  def GetDatasets(self):
    np.random.seed(self.random_seed)

    if not self.discrete_true_mass:
      true_mass = np.random.uniform(self.true_mass_ranges[0], self.true_mass_ranges[1], size=self.array_size)
    else:
      true_mass = np.random.choice(self.discrete_true_masses, size=self.array_size)
    if not self.discrete_sig_frac:
      sig_frac = np.random.uniform(self.sig_frac_ranges[0], self.sig_frac_ranges[1], size=self.array_size)
    else:
      if not self.fix_bkg_events:
        sig_frac = np.random.choice(self.discrete_sig_fracs, size=self.array_size)
      else:
        probs = [1/(1-s) for s in self.discrete_sig_fracs]
        norm_probs = [p/sum(probs) for p in probs]
        sig_frac = np.random.choice(self.discrete_sig_fracs, size=self.array_size, p=norm_probs)

    true_mass = true_mass.reshape((len(true_mass),1))
    sig_frac = sig_frac.reshape((len(sig_frac),1))
    if self.return_sig_frac:
      context = np.concatenate((true_mass, sig_frac), axis=1)
    else:
      context = true_mass
    X = np.array([self.RandomSignalPlusBackgroundEvent(true_mass[ind][0],sig_frac[ind][0]) for ind in range(len(true_mass))])
    X = X.reshape(len(X),1)
    return_dict = {
      "X" : X,
      "Y" : context,
    }
    return return_dict

### Make train, test and validation datasets ###
md = MakeDatasets()
if args.use_signal_fraction:
  md.discrete_sig_frac = args.use_discrete_signal_fraction
  md.return_sig_frac = True
elif args.add_background:
  md.discrete_sig_frac = True
  md.discrete_sig_fracs = [args.signal_fraction]
else:
  md.discrete_sig_frac = True
  md.discrete_sig_fracs = [1.0]
md.discrete_true_mass = args.use_discrete_true_mass

train_config = {"random_seed" : 24}
md.SetParameters(train_config)
train = md.GetDatasets()

test_config = {"random_seed" : 25}
md.SetParameters(test_config)
test = md.GetDatasets()

val_config = {
  "fix_bkg_events" : args.use_signal_fraction,
  "random_seed" : 26, 
  "discrete_true_mass" : True,
  "discrete_true_masses" : [float(i) for i in args.plot_true_masses.split(",")],
  "array_size" : 10**7
  }
if args.use_signal_fraction:
  val_config["discrete_sig_frac"] = True,
  val_config["discrete_sig_fracs"] = [float(i) for i in args.plot_signal_fractions.split(",")]
md.SetParameters(val_config)
val = md.GetDatasets()

data = DataProcessor(
  {
    "train" : train,
    "test" : test,
    "val" : val,
  }
)
data.standardise_data = {"X":True,"Y":True}

### Plot initial inputs ###
if not args.skip_initial_distribution:
  data.plot_dir = f"plots/{name}"
  data.PlotUniqueY(data_key="val")

### Set up model and training parameters ###
if args.add_background:
  parameters = {
    "num_coupling_layer" : 8,
    "units_per_coupling_layer" : 128,
    "num_dense_layers" : 4,
    "activation" : "relu",
    "dropout" : True,
    "mc_dropout" : True,
    "summary_dimensions" : 1, 
    "epochs" : 15,
    "batch_size" : 2**6, 
    "early_stopping" : True,
    "learning_rate" : 1e-3, 
    "coupling_design" : "interleaved", 
    "permutation" : "learnable", 
    "decay_rate" : 0.5,
    }
else:
  parameters = {
    "num_coupling_layers" : 8,
    "units_per_coupling_layer" : 128,
    "num_dense_layers" : 1,
    "activation" : "relu",
    "dropout" : True,
    "mc_dropout" : False,
    "summary_dimensions" : 1,
    "epochs" : 10,
    "batch_size" : 2**6,
    "early_stopping" : True,
    "learning_rate" : 1e-3,
    "coupling_design" : "affine",
    "permutation" : "learnable",
    "decay_rate" : 0.5,
    }

model = Model(data, options=parameters)
model.plot_dir = f"plots/{name}"
print("- Building model")
model.BuildModel()

if not args.load_model:
  ### Run training ###
  print("- Running training")
  model.BuildTrainer()
  model.Train()
  model.Save(name=f"models/{name}.h5")
else:
  ### Load in weights ###
  print("- Loading weights")
  model.Load(name=f"models/{name}.h5")

### Do validation steps ###
inference = Inference(model, data)
inference.plot_dir = f"plots/{name}"

### Test closure by applying train and test context back and using it as a generator ###
if not args.skip_closure:
  print("- Plotting closure")
  inference.PlotClosure()

### Test closure by using it as a generator for individual context values ###
if not args.skip_generation:
  print("- Plotting generated distributions")
  inference.PlotGeneration()

### Make a plot of the probabilities of a true value when varying the data also with the sampled density ###
if not args.skip_probability:
  print("- Making probability plots")
  inference.PlotProbAndSampleDensityFor1DX()
  
### Get best fit and draw likelihoods ###
if not args.skip_inference:
  print("- Getting the best fit value from a single toy and drawing likelihood scans")
  if args.use_signal_fraction:
    initial_guess = np.array([172.5,0.2])
  else:
    initial_guess = np.array([172.5])

  inference.n_events_in_toy=1000
  inference.GetBestFit(initial_guess)
  inference.GetProfiledIntervals()
  inference.PrintBestFitAndIntervals()
  inference.Draw1DLikelihoods()

  if not args.skip_comparison:
    inference.DrawComparison()

  if args.use_signal_fraction:
    inference.Draw2DLikelihoods()