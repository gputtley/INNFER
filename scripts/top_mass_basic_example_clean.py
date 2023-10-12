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

There are also sub options for runnings for using discrete input values and for
changing the input signal fractions, in the case of 2.
"""

print("- Importing packages")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import copy
import argparse
import time
import bayesflow as bf
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import chisquare
from plotting import plot_likelihood, plot_histogram_with_ratio, plot_2d_likelihood
from scipy.optimize import minimize
from scipy.integrate import simpson

tf.random.set_seed(42) 
tf.keras.utils.set_random_seed(42)
hep.style.use("CMS")

parser = argparse.ArgumentParser()
parser.add_argument('--use-summary-network', help= 'Use a summary network',  action='store_true')
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

args = parser.parse_args()

if args.use_signal_fraction:
  args.add_background = True

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

# Hyperparameters for networks
if args.add_background:
  hyperparameters = {
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
  latent_dim = 2
else:
  hyperparameters = {
    "num_coupling_layer" : 8,
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
  latent_dim = 1

### Data generation ###
print("- Making datasets")

class MakeDatasets():

  def __init__(self):
    self.true_mass_ranges = [165.0,180.0]
    self.sig_frac_ranges = [0.0,0.5]
    self.true_mass_fixed_value = None
    self.sig_frac_fixed_value = None
    self.array_size = 10**6

    self.norm_true_mass = False
    self.norm_sig_frac = False
    self.norm_X = False

    self.true_mass_mean = None
    self.true_mass_std = None
    self.sig_frac_mean = None
    self.sig_frac_std = None   
    self.X_mean = None
    self.X_std = None 

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

    self.sig_res = 0.01
    self.bkg_lambda = 0.1
    self.bkg_const = 160.0
    self.bkg_ranges = [160.0,185.0]

    self.return_sig_frac = False
    self.for_training = False
    self.for_inference = False
    self.using_summary = False

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

  def Normalise(self, x, mean, std):
    if mean == None or std == None:
      if x.std() != 0.0:
        return ((x-x.mean())/x.std()).astype(np.float32), x.mean(), x.std()
      else:
        return np.zeros(len(x)), x.mean(), x.std()
    else:
      if std != 0.0:
        return ((x-mean)/std).astype(np.float32), mean, std
      else:
        return np.zeros(len(x)), mean, std

  def GetDatasets(self, only_context=False):

    np.random.seed(self.random_seed)

    if self.true_mass_fixed_value != None:
      true_mass = self.true_mass_fixed_value*np.ones(self.array_size)
    elif not self.discrete_true_mass:
      true_mass = np.random.uniform(self.true_mass_ranges[0], self.true_mass_ranges[1], size=self.array_size)
    else:
      true_mass = np.random.choice(self.discrete_true_masses, size=self.array_size)

    if self.sig_frac_fixed_value != None:
      sig_frac = self.sig_frac_fixed_value*np.ones(self.array_size)
    elif not self.discrete_sig_frac:
      sig_frac = np.random.uniform(self.sig_frac_ranges[0], self.sig_frac_ranges[1], size=self.array_size)
    else:
      sig_frac = np.random.choice(self.discrete_sig_fracs, size=self.array_size)

    if not only_context:
      X = np.array([self.RandomSignalPlusBackgroundEvent(true_mass[ind],sig_frac[ind]) for ind in range(len(true_mass))])

    if self.norm_true_mass:
      true_mass, self.true_mass_mean, self.true_mass_std = self.Normalise(true_mass, self.true_mass_mean, self.true_mass_std)
    if self.norm_sig_frac:
      sig_frac, self.sig_frac_mean, self.sig_frac_std = self.Normalise(sig_frac, self.sig_frac_mean, self.sig_frac_std)
    if self.norm_X and not only_context:
      X, self.X_mean, self.X_std = self.Normalise(X, self.X_mean, self.X_std)

    if not only_context:
      X = X.reshape((len(X),1))
    true_mass = true_mass.reshape((len(true_mass),1))
    sig_frac = sig_frac.reshape((len(sig_frac),1))

    if self.return_sig_frac:
      context = np.concatenate((true_mass, sig_frac), axis=1)
    else:
      context = true_mass

    if self.using_summary:
      context = context.reshape((context.shape[0],context.shape[1],1))
    
    return_dict = {}
    if self.for_training:
      if not only_context:
        return_dict["prior_draws"] = X
      return_dict["sim_data"] = context
    elif self.for_inference:
      if not only_context:
        return_dict["parameters"] = X
      if self.using_summary:
        return_dict["summary_conditions"] = context
      else:
        return_dict["direct_conditions"] = context
    else:
      return_dict["true_mass"] = true_mass
      return_dict["sig_frac"] = sig_frac
      if not only_context:
        return_dict["X"] = X

    return return_dict

md = MakeDatasets()

if args.add_background and args.use_signal_fraction:
  sig_frac_fixed_value = None
elif args.add_background:
  sig_frac_fixed_value = args.signal_fraction
else:
  sig_frac_fixed_value = 1.0

train_config = {
  "for_training" : True,
  "random_seed" : 24,
  "norm_X" : True,
  "norm_true_mass" : True, 
  "norm_sig_frac" : True,
  "sig_frac_fixed_value" : sig_frac_fixed_value,
  "return_sig_frac" : args.use_signal_fraction,
  "discrete_true_mass" : args.use_discrete_true_mass,
  "discrete_sig_frac" : args.use_discrete_signal_fraction,
}
md.SetParameters(train_config)
train = md.GetDatasets()

test_config = {
  "random_seed" : 42,
}
md.SetParameters(test_config)
test = md.GetDatasets()

### Set up models ###
print("- Setting up models")

if hyperparameters["coupling_design"] == "interleaved":
  settings = {
    "affine" : {
      "dense_args": dict(
        kernel_regularizer=None, 
        units=hyperparameters["units_per_coupling_layer"], 
        activation=hyperparameters["activation"]), 
      "dropout": hyperparameters["dropout"],
      "num_dense": hyperparameters["num_dense_layers"],      
    },
    "spline" : {
      "dense_args": dict(
        kernel_regularizer=None, 
        units=hyperparameters["units_per_coupling_layer"], 
        activation=hyperparameters["activation"]), 
      "dropout": hyperparameters["dropout"],
      "num_dense": hyperparameters["num_dense_layers"],      
    },
  }
else:
  settings = {
    "dense_args": dict(
      kernel_regularizer=None, 
      units=hyperparameters["units_per_coupling_layer"], 
      activation=hyperparameters["activation"]), 
    "dropout": hyperparameters["dropout"],
    "mc_dropout": hyperparameters["mc_dropout"],
    "num_dense": hyperparameters["num_dense_layers"],
    }

inference_net = bf.networks.InvertibleNetwork(
  num_params=latent_dim,
  num_coupling_layers=hyperparameters["num_coupling_layer"],
  permutation=hyperparameters["permutation"],
  coupling_design=hyperparameters["coupling_design"],
  coupling_settings=settings
  )

if args.use_summary_network:
  summary_net = bf.networks.DeepSet(summary_dim=hyperparameters["summary_dimensions"]) 
  amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
else:
  amortizer = bf.amortizers.AmortizedPosterior(inference_net)

# Define here because useful for a lot of the next steps
if args.only_infer_true_mass == None:
  true_mass_plot = np.linspace(171.0, 174.0, num=4, endpoint=True)
else:
  true_mass_plot = [args.only_infer_true_mass]
if args.only_infer_signal_fraction == None:
  if args.use_signal_fraction:
    sig_frac_plot = np.linspace(0.1, 0.3, num=3, endpoint=True)
  elif not args.add_background:
    sig_frac_plot = np.array([1.0])
  else:
    sig_frac_plot = np.array([args.signal_fraction])
else:
  sig_frac_plot = [args.only_infer_signal_fraction]

### Plot initial inputs ##
if not args.skip_initial_distribution:
  print("- Plotting initial inputs")

  # Get binning
  binning_config = {
    "random_seed" : 24,
    "array_size" : 100000,
    "norm_true_mass" : False,
    "norm_sig_frac" : False,
    "norm_X" : False,
    "for_training" : False,
    "true_mass_ranges" : [true_mass_plot[0],true_mass_plot[-1]],
    "sig_frac_ranges" : [sig_frac_plot[0],sig_frac_plot[-1]],
  }
  md.SetParameters(binning_config)
  td = md.GetDatasets()["X"]
  _, bins = np.histogram(td,bins=40)

  fig, ax = plt.subplots()
  hep.cms.text("Work in progress",ax=ax)
  for tm in true_mass_plot:
    for sf in sig_frac_plot:

      tm = round(tm,1)
      sf = round(sf,1)

      plotting_config = {
        "array_size" : int(round(1000000/(1-sf))) if sf != 1 else 1000000, # make sure we always get the same number of background events
        "true_mass_fixed_value" : tm,
        "sig_frac_fixed_value" : sf,
      }
      md.SetParameters(plotting_config)
      if args.use_signal_fraction:
        label = "y=({},{})".format(tm,sf)
      else:
        label = "y={}".format(tm)
      plt.plot(bins[:-1], np.histogram(md.GetDatasets()["X"],bins=bins)[0], label=label)

  plt.xlabel("x")
  plt.ylabel('Events')
  plt.legend()
  plt.tight_layout()
  plt.savefig("plots/{}/{}_initial_distributions.pdf".format(name,name))
  print("Created plots/{}/{}_initial_distributions.pdf".format(name,name))

### Run training or load the model in from file ###
if not args.load_model:
  # Set up trainer
  print("- Training model")
  if args.use_summary_network: 
    config = None
  else:
    def config(forward_dict):
      out_dict = {}
      out_dict["direct_conditions"] = forward_dict["sim_data"]
      out_dict["parameters"] = forward_dict["prior_draws"]
      return out_dict
    
  trainer = bf.trainers.Trainer(amortizer=amortizer, configurator=config, default_lr=hyperparameters["learning_rate"], memory=False)
  #trainer = TrainerWithWeights(amortizer=amortizer, configurator=config, default_lr=hyperparameters["learning_rate"], memory=False)

  # Train model
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    trainer.default_lr, 
    decay_steps=int(len(train["prior_draws"])/hyperparameters["batch_size"]), 
    decay_rate=hyperparameters["decay_rate"]
  )

  if latent_dim == 2:
    train["prior_draws"] = np.column_stack((train["prior_draws"].flatten(), np.random.normal(0.0, 1.0, (len(train["prior_draws"].flatten()),))))
    test["prior_draws"] = np.column_stack((test["prior_draws"].flatten(), np.random.normal(0.0, 1.0, (len(test["prior_draws"].flatten()),))))
    #train["prior_draws"] = np.column_stack((train["prior_draws"].flatten(), np.zeros(len(train["prior_draws"].flatten()))))
    #test["prior_draws"] = np.column_stack((test["prior_draws"].flatten(), np.zeros(len(test["prior_draws"].flatten()))))


  history = trainer.train_offline(
    train, 
    epochs=hyperparameters["epochs"], 
    batch_size=hyperparameters["batch_size"], 
    validation_sims=test, 
    early_stopping=hyperparameters["early_stopping"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
  )
  
  if latent_dim == 2:
    train["prior_draws"] = train["prior_draws"][:,0].reshape(len(train["prior_draws"]),1)
    test["prior_draws"] = test["prior_draws"][:,0].reshape(len(test["prior_draws"]),1)


  # Plot the loss
  f = bf.diagnostics.plot_losses(history["train_losses"], history["val_losses"])
  f.savefig("plots/{}/{}_loss.pdf".format(name,name))

  # Save the weights to file
  inference_net.save_weights("models/inference_net_{}.h5".format(name))
  if args.use_summary_network:
    summary_net.save_weights("models/summary_net_{}.h5".format(name))

else:

  # Load model in from file
  print("- Loading model")
  if not args.use_summary_network:
    _ = inference_net(train["prior_draws"],train["sim_data"])
    inference_net.load_weights("models/inference_net_{}.h5".format(name))
  else:
    sum = summary_net(train["sim_data"])
    summary_net.load_weights("models/summary_net_{}.h5".format(name))
    _ = inference_net(train["prior_draws"],sum)
    inference_net.load_weights("models/inference_net_{}.h5".format(name))

### Test closure by using it to generate the dataset it was given ###
if not args.skip_closure:
  print("- Plotting closure distributions")

  # Make the histograms
  train_hist, train_bins = np.histogram(train["prior_draws"],bins=40)
  test_hist, test_bins = np.histogram(test["prior_draws"],bins=40)

  # Make synthetic dataset
  train_amort = {}
  test_amort = {}
  if args.use_summary_network:
    key = "summary_conditions"
  else:
    key = "direct_conditions"
  train_amort[key] = train["sim_data"].reshape(train["sim_data"].shape[0],train["sim_data"].shape[1])
  test_amort[key] = test["sim_data"].reshape(test["sim_data"].shape[0],test["sim_data"].shape[1])
  train_synth = amortizer.sample(train_amort, 1)[:,0,0]
  test_synth = amortizer.sample(test_amort, 1)[:,0,0]
  synth_train_hist, _ = np.histogram(train_synth,bins=train_bins)
  synth_test_hist, _ = np.histogram(test_synth,bins=test_bins)

  # Plot synthetic dataset against the simulated dataset
  plot_histogram_with_ratio(
    train_hist, 
    synth_train_hist, 
    train_bins, 
    name_1='Simulated', 
    name_2='Synthetic',
    xlabel="x",
    name="plots/{}/{}_closure_train".format(name,name), 
    title_right = "Train",
    density = True,
    use_stat_err = True,
    )

  plot_histogram_with_ratio(
    test_hist, 
    synth_test_hist, 
    test_bins, 
    name_1='Simulated', 
    name_2='Synthetic',
    xlabel="x",
    name="plots/{}/{}_closure_test".format(name,name), 
    title_right = "Test",
    density = True,
    use_stat_err = True,
    )

### Test closure by using it as a generator ###
if not args.skip_generation:
  print("- Plotting generated distributions")

  sum_chi2 = 0.0
  for tm in true_mass_plot:
    for sf in sig_frac_plot:

      tm = round(tm,1)
      sf = round(sf,1)

      generator_config = {
        "array_size" : 100000,
        "true_mass_fixed_value" : tm,
        "sig_frac_fixed_value" : sf,
        "for_training" : False,
        "for_inference" : True,
        "norm_true_mass" : True,
        "norm_sig_frac" : True,
        "norm_X" : True,
      }
      md.SetParameters(generator_config)
      inf_data = md.GetDatasets(only_context=True)

      # Make synthetic dataset
      synth = (amortizer.sample(inf_data, 1)[:,0,0]*md.X_std) + md.X_mean

      # Get the simulated dataset
      sim_config = {
        "array_size" : 100000,
        "true_mass_fixed_value" : tm,
        "sig_frac_fixed_value" : sf,
        "norm_true_mass" : False,
        "norm_sig_frac" : False,
        "norm_X" : False,
        "for_inference" : False,
      }
      md.SetParameters(sim_config)
      sim = md.GetDatasets()["X"]

      # Make the histograms
      sim_hist, bins = np.histogram(sim,bins=40)
      synth_hist, _ = np.histogram(synth,bins=bins)
      
      if args.use_signal_fraction:
        label = "y=({},{})".format(tm,sf)
        extra_name = "{}_{}".format(str(tm).replace(".","p"),str(sf).replace(".","p"))
      else:
        label = "y={}".format(tm)
        extra_name = str(tm).replace(".","p")

      # Plot synthetic dataset against the simulated dataset
      plot_histogram_with_ratio(
        sim_hist, 
        synth_hist, 
        bins, 
        name_1='Simulated', 
        name_2='Synthetic',
        xlabel="x",
        name="plots/{}/{}_synthetic_{}".format(name,name,extra_name), 
        title_right = label,
        density = True,
        use_stat_err = True,
        )
      non_zero_indices = np.intersect1d(np.where(sim_hist > 0.0),np.where(synth_hist > 0.0))
      sim_hist = sim_hist[non_zero_indices]
      synth_hist = synth_hist[non_zero_indices]
      norm_sim_hist = sim_hist/np.sum(sim_hist)
      norm_synth_hist = synth_hist/np.sum(synth_hist)
      chi2, pvalue = chisquare(f_obs=norm_sim_hist, f_exp=norm_synth_hist)
      sum_chi2 += chi2
      print("Chi-squared = ", chi2)
  print("Average chi-squared = ", sum_chi2/(len(true_mass_plot)*len(sig_frac_plot)))
      

def GetProb(true_mass, sig_frac, data, integral_bins=100, log=False):
  if latent_dim == 1:
    prob_config = {
      "array_size" : len(data),
      "true_mass_fixed_value" : true_mass,
      "sig_frac_fixed_value" : sig_frac,
      "for_training" : False,
      "for_inference" : True,
      "norm_true_mass" : True,
      "norm_sig_frac" : True,
      "norm_X" : True,
    }
    md.SetParameters(prob_config)
    inf_data = md.GetDatasets(only_context=True) 
    inf_data["parameters"] = md.Normalise(data.reshape((len(data),1)), md.X_mean, md.X_std)[0]
    tf.random.set_seed(42)
    prob = np.exp(amortizer.log_posterior(inf_data))/md.X_std

  elif latent_dim == 2:
    #start_time = time.time()
    prob_config = {
      "array_size" : len(data)*integral_bins,
      "true_mass_fixed_value" : true_mass,
      "sig_frac_fixed_value" : sig_frac,
      "for_training" : False,
      "for_inference" : True,
      "norm_true_mass" : True,
      "norm_sig_frac" : True,
      "norm_X" : True,
    }
    md.SetParameters(prob_config)
    inf_data = md.GetDatasets(only_context=True)
    norm_data = md.Normalise(data, md.X_mean, md.X_std)[0]
    gauss = np.linspace(-5, 5, integral_bins)
    x2, x1 = np.meshgrid(gauss, norm_data)
    inf_data["parameters"] = np.column_stack((x1.ravel(), x2.ravel()))
    tf.random.set_seed(42)
    p = np.exp(amortizer.log_posterior(inf_data))/md.X_std
    prob = np.array([])
    for i in range(len(data)):
      start_idx = i * integral_bins
      end_idx = (i + 1) * integral_bins
      prob = np.append(prob,simpson(p[start_idx:end_idx], gauss))

    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Elapsed time for integrating: {elapsed_time} seconds")

  if log:
    return np.log(prob)
  else:
    return prob
    
### Need to integrate the pdf and ensure the integral is 1 ###
def GetNormProb(true_mass, sig_frac, data, integral_range=[145.0,200.0], integral_bins=200, log=False, norm=True):
  int_data = np.linspace(integral_range[0],integral_range[1],num=integral_bins)
  p = GetProb(true_mass, sig_frac, int_data)
  integral = simpson(p, int_data)
  prob = GetProb(true_mass, sig_frac, data)/integral
  if log:
    return np.log(prob)
  else:
    return prob
    
### Make a plot of the probabilities of a true value when varying the data also with the sampled density ###
if not args.skip_probability:
  print("- Making probability plots")
  x_plot = np.linspace(155.0,190.0,num=100)
  for tm in true_mass_plot:
    for sf in sig_frac_plot:
      tm = round(tm,1)
      sf = round(sf,1)
      #prob = GetNormProb(tm, sf, x_plot[:-1])
      prob = GetProb(tm, sf, x_plot[:-1])
      print(" - Integral =", simpson(prob,x_plot[:-1]))
      fig, ax = plt.subplots()
      hep.cms.text("Work in progress",ax=ax)
      plt.plot(x_plot[:-1], prob, linestyle='-', color="blue", label="Probability")
      generator_config = {
        "array_size" : 100000,
        "true_mass_fixed_value" : tm,
        "sig_frac_fixed_value" : sf,
        "for_training" : False,
        "for_inference" : True,
        "norm_true_mass" : True,
        "norm_sig_frac" : True,
        "norm_X" : True,
      }
      md.SetParameters(generator_config)
      inf_data = md.GetDatasets(only_context=True)
      synth = (amortizer.sample(inf_data, 1)[:,0,0]*md.X_std) + md.X_mean
      synth_hist,synth_bins = np.histogram(synth,bins=x_plot,density=True)

      bin_centers = [i+(x_plot[1]-x_plot[0])/2 for i in x_plot[:-1]]
      plt.plot(bin_centers, synth_hist, linestyle='-', color="red", label="Sampled Density")

      plt.legend()

      plt.xlabel("x")
      if args.use_signal_fraction:
        label = "{},{}".format(tm,sf)
        extra_name = "{}_{}".format(str(tm).replace(".","p"),str(sf).replace(".","p"))
      else:
        label = "{}".format(tm)
        extra_name = str(tm).replace(".","p")
      plt.ylabel('p(x|{})'.format(label))
      plt.tight_layout()
      plt.savefig("plots/{}/{}_probability_{}.pdf".format(name,name,extra_name))
      print("Created plots/{}/{}_probability_{}.pdf".format(name,name,extra_name))
      plt.close()
        

### Get best fit and draw likelihoods ###
if not args.skip_inference:
  print("- Getting the best fit value and drawing likelihood scans")

  # Number of events for scan
  test_array_size = int(round(1000/args.signal_fraction))

  # Inputs for minimisation
  if not args.use_signal_fraction:
    initial_guess = 172.5
    par_ranges = [170,175]
  else:
    initial_guess = [172.5,0.1]
    par_ranges = [[170,175],[0.0,0.4]]

  # Function to get -2deltaLL from a tested context value for a given true value
  def objective_function(params, data, shift=0, absolute=False, par_range=None, freeze_sig_frac=None, freeze_true_mass=None):
    #start_time = time.time()
    #print(params,freeze_sig_frac,freeze_true_mass)
    if freeze_sig_frac != None:
      true_mass = params
      sig_frac = freeze_sig_frac
      if par_range != None:
        if not (params > par_range[0] and params < par_range[1]):
          return np.inf
    elif freeze_true_mass != None:
      true_mass = freeze_true_mass
      sig_frac = params
      if par_range != None:
        if not (params > par_range[0] and params < par_range[1]):
          return np.inf
    else:
      true_mass = params[0]
      sig_frac = params[1]
      for i in [0,1]:
        if par_range[i] != None:
          if not (params[i] > par_range[i][0] and params[i] < par_range[i][1]):
            return np.inf
      
    #mtnll = -2*GetNormProb(true_mass, sig_frac, data, log=True).sum()
    mtnll = -2*GetProb(true_mass, sig_frac, data, log=True).sum()
    mtnll -= shift
    if absolute: mtnll = abs(mtnll)
    #print(true_mass, sig_frac, params, mtnll)
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Elapsed time: {elapsed_time} seconds")
    return mtnll

  for tm_ind, tm in enumerate(true_mass_plot):
    for sf in sig_frac_plot:

      print(" - Testing the true values ({},{})".format(tm,sf))
      # Make data
      data_config = {
        "array_size" : test_array_size,
        "true_mass_fixed_value" : tm,
        "sig_frac_fixed_value" : sf,
        "for_training" : False,
        "for_inference" : False,
        "norm_true_mass" : True,
        "norm_sig_frac" : True,
        "norm_X" : False,
        "random_seed" : np.random.RandomState().randint(0, 100),
      }
      md.SetParameters(data_config)
      data = md.GetDatasets()
      data = data["X"]

      minimise_options = {'xatol': 0.0001, 'fatol': 0.0001, 'maxiter': 30}
      minimise_tolerence = 0.1

      #print(initial_guess)
      # Get best fit
      print("   - Getting best fit")
      result = minimize(objective_function, initial_guess, method='Nelder-Mead', args=(data,0,False,par_ranges,None if args.use_signal_fraction else sf,None), tol=minimise_tolerence, options=minimise_options)

      # Draw 1d likelihoods, profile if needed
      if not args.use_signal_fraction:
        profile = [None]
        true_extra_name = str(tm).replace(".","p")
        y_name = str(tm)
      else:
        profile = ["sig_frac","true_mass"]
        true_extra_name = str(tm).replace(".","p") + "_" + str(sf).replace(".","p")
        y_name = "({},{})".format(round(tm,2),round(sf,2))

      crossings_for_2d = {}
      for prof in profile:
      
        freeze_sig_frac = None
        freeze_true_mass = None
        if prof == "sig_frac":
          prof_extra_name = "_vary_true_mass"
          vary_ind = 0
          freeze_sig_frac = result.x[1]
          true_value = tm
          xlabel = "top mass"
        elif prof == "true_mass":
          prof_extra_name = "_vary_sig_frac"
          vary_ind = 1
          freeze_true_mass = result.x[0]
          true_value = sf
          xlabel = "signal fraction"
        elif prof == None:
          prof_extra_name = ""
          vary_ind = 0
          freeze_sig_frac = sf
          true_value = tm
          xlabel = "top mass"

        crossings = {
          0 : result.x[vary_ind]
        }        

        if not prof == None:
          print("   - Profiling {}".format(prof))
      
        for sigma in [-1,1]:
          # Set ranges to make sure we are getting the up or down crossing and profile if required
          if prof == None:
            if sigma < 0:
              prof_par_ranges = [par_ranges[0],result.x[0]]
            else:
              prof_par_ranges = [result.x[0],par_ranges[1]]
          else:
            if sigma < 0:
              prof_par_ranges = [par_ranges[vary_ind][0],result.x[vary_ind]]
            else:
              prof_par_ranges = [result.x[vary_ind],par_ranges[vary_ind][1]]
          prof_initial_guess = result.x[vary_ind]

          # Get crossings
          print("   - Getting crossings for {}".format(sigma))
          crossings[sigma] = minimize(objective_function, prof_initial_guess, method='Nelder-Mead', args=(data,result.fun+(sigma**2),True,prof_par_ranges,freeze_sig_frac,freeze_true_mass), tol=minimise_tolerence, options=minimise_options).x[0]
        crossings_for_2d[prof] = copy.deepcopy(crossings)

        # Print values
        dp = 3
        if crossings[0] > true_value:
          approx_deviation = (crossings[0] - true_value)/(crossings[0]-crossings[-1])
        else:
          approx_deviation = (true_value - crossings[0])/(crossings[1]-crossings[0])
        print("   - True Value: {}, Calculated Value: {} + {} - {}, Approx. deviation: {}".format(true_value,round(crossings[0],dp),round(crossings[1]-crossings[0],dp),round(crossings[0]-crossings[-1],dp),round(approx_deviation,dp)))

        # Calculate likelihood points
        n_points = 20
        estimated_sigma_shown = 3
        lower_plot = crossings[0]-(estimated_sigma_shown*(crossings[0]-crossings[-1]))
        higher_plot = crossings[0]+(estimated_sigma_shown*(crossings[1]-crossings[0]))
        plot_poi = np.linspace(lower_plot,higher_plot,num=n_points+1, endpoint=True)
        plot_m2dlls = []
        for poi in plot_poi:
          if prof == None:
            c = [poi,sf]
          else:
            c = [result.x[0],result.x[1]]
            c[vary_ind] = poi
          plot_m2dlls.append(objective_function(c, data, shift=result.fun, par_range=[None,None]))

        #for ind in range(len(plot_poi)):
        #  print(plot_poi[ind],plot_m2dlls[ind])

        # Draw likelihood
        plot_likelihood(
          plot_poi, 
          plot_m2dlls, 
          crossings, 
          name="plots/{}/{}_likelihood_{}".format(name,name,true_extra_name+prof_extra_name), 
          xlabel=xlabel, 
          true_value=true_value,
          cap_at=9
          )

      if not args.skip_comparison:
        print("   - Making comparison plots")

        data_hist, bins = np.histogram(data,bins=40)

        true_dist_config = {
          "array_size" : 100000,
          "true_mass_fixed_value" : tm,
          "sig_frac_fixed_value" : sf,
          "for_training" : False,
          "for_inference" : False,
          "norm_true_mass" : False,
          "norm_sig_frac" : False,
          "norm_X" : False,
          "random_seed" : 42,
        }
        md.SetParameters(true_dist_config)
        true_dist = md.GetDatasets()["X"]
        true_hist = np.histogram(true_dist,bins=bins)[0]

        infer_dist_config = {
          "array_size" : 100000,
          "true_mass_fixed_value" : result.x[0],
          "sig_frac_fixed_value" : result.x[1] if args.use_signal_fraction else sf,
          "for_training" : False,
          "for_inference" : False,
          "norm_true_mass" : False,
          "norm_sig_frac" : False,
          "norm_X" : False,
          "random_seed" : 42,
        }
        md.SetParameters(infer_dist_config)
        infer_dist = md.GetDatasets()["X"]
        infer_hist = np.histogram(infer_dist,bins=bins)[0]

        gen_true_dist_config = {
          "array_size" : 100000,
          "true_mass_fixed_value" : tm,
          "sig_frac_fixed_value" : sf,
          "for_training" : False,
          "for_inference" : True,
          "norm_true_mass" : True,
          "norm_sig_frac" : True,
          "norm_X" : False,
          "only_context" : True,
          "random_seed" : 42,
        }
        md.SetParameters(gen_true_dist_config)
        gen_true_dist = md.GetDatasets()
        synth_true_dist = (amortizer.sample(gen_true_dist, 1)[:,0,0]*md.X_std) + md.X_mean
        synth_true_hist = np.histogram(synth_true_dist,bins=bins)[0]

        gen_infer_dist_config = {
          "array_size" : 100000,
          "true_mass_fixed_value" : result.x[0],
          "sig_frac_fixed_value" : result.x[1] if args.use_signal_fraction else sf,
          "for_training" : False,
          "for_inference" : True,
          "norm_true_mass" : True,
          "norm_sig_frac" : True,
          "norm_X" : False,
          "only_context" : True,
          "random_seed" : 42,
        }
        md.SetParameters(gen_infer_dist_config)
        gen_infer_dist = md.GetDatasets()
        synth_infer_dist = (amortizer.sample(gen_infer_dist, 1)[:,0,0]*md.X_std) + md.X_mean
        synth_infer_hist = np.histogram(synth_infer_dist,bins=bins)[0]

        fig, ax = plt.subplots()
        hep.cms.text("Work in progress",ax=ax)

        if not args.use_signal_fraction:
          y_bf_name = str(round(result.x[0],2))
        else:
          y_bf_name = "({},{})".format(round(result.x[0],2),round(result.x[1],2))

        plt.plot(bins[:-1], true_hist/sum(true_hist), label="True y={}".format(y_name), linestyle='-', color='red')
        plt.plot(bins[:-1], infer_hist/sum(infer_hist), label="True y={}".format(y_bf_name), linestyle='-', color='blue')
        plt.plot(bins[:-1], synth_true_hist/sum(synth_true_hist), label="Synth y={}".format(y_name), linestyle='--', color='red')
        plt.plot(bins[:-1], synth_infer_hist/sum(synth_infer_hist), label="Synth y={}".format(y_bf_name), linestyle='--', color='blue')

        data_hist_err = np.sqrt(data_hist)
        plt.errorbar(bins[:-1], data_hist/sum(data_hist), yerr=data_hist_err/sum(data_hist), label="Data y={}".format(y_name), markerfacecolor='none', linestyle='None', fmt='k+')

        plt.xlabel("x")
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/{}/{}_inferred_distributions_{}.pdf".format(name,name,true_extra_name))
        plt.close()
        print("Created plots/{}/{}_inferred_distributions_{}.pdf".format(name,name,true_extra_name))

      # Draw 2d likelihoods
      if args.use_signal_fraction:
        print(" - Drawing 2d likelihood")

        n_points = 20
        estimated_sigma_shown = 4
        x_lower_plot = crossings_for_2d['sig_frac'][0]-(estimated_sigma_shown*(crossings_for_2d['sig_frac'][0]-crossings_for_2d['sig_frac'][-1]))
        x_higher_plot = crossings_for_2d['sig_frac'][0]+(estimated_sigma_shown*(crossings_for_2d['sig_frac'][1]-crossings_for_2d['sig_frac'][0]))
        y_lower_plot = crossings_for_2d['true_mass'][0]-(estimated_sigma_shown*(crossings_for_2d['true_mass'][0]-crossings_for_2d['true_mass'][-1]))
        y_higher_plot = crossings_for_2d['true_mass'][0]+(estimated_sigma_shown*(crossings_for_2d['true_mass'][1]-crossings_for_2d['true_mass'][0]))

        x_range = np.linspace(x_lower_plot,x_higher_plot,num=n_points)
        y_range = np.linspace(y_lower_plot,y_higher_plot,num=n_points)

        vals = []
        for yind, yp in enumerate(y_range):
          vals.append([objective_function([xp,yp], data, shift=result.fun, par_range=[None,None]) for xp in x_range])

        plot_2d_likelihood(
          x_range, 
          y_range, 
          vals, 
          name="plots/{}/{}_2d_likelihood_{}".format(name,name,true_extra_name), 
          xlabel="top mass", 
          ylabel="signal fraction", 
          best_fit=[result.x[0],result.x[1]], 
          true_value=[tm,sf], 
          title_right="y="+y_name
          )
