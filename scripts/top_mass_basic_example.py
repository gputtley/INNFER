"""
Title: Top mass basic example:
Author: George Uttley

This is a simple example where we try and infer the mass of the top quark from one reconstructed mass like context variable using the bayesflow package.
"""

print("- Importing packages")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import bayesflow as bf
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import tensorflow as tf
from plotting import plot_likelihood, plot_histogram_with_ratio
from scipy.optimize import minimize

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
hep.style.use("CMS")

parser = argparse.ArgumentParser()
parser.add_argument('--use-summary-network', help= 'Use a summary network',  action='store_true')
parser.add_argument('--use-discrete', help= 'Use discrete true values',  action='store_true')
parser.add_argument('--add-background', help= 'Add a falling background',  action='store_true')
parser.add_argument('--load-model', help= 'Load model from file',  action='store_true')
parser.add_argument('--skip-initial-distribution', help= 'Skip making the initial distribution plots',  action='store_true')
parser.add_argument('--skip-generation', help= 'Skip making the generation plots',  action='store_true')
parser.add_argument('--skip-probability', help= 'Skip draw the probability',  action='store_true')
parser.add_argument('--skip-inference', help= 'Skip performing the inference',  action='store_true')
parser.add_argument('--extend-likelihood-range', help= 'Extend the plotted likelihood range',  action='store_true')

args = parser.parse_args()

name = "top_mass_basic_example"
if args.add_background:
  name += "_with_bkg"

# Hyperparameters for networks
if args.add_background:
  hyperparameters = {
    "num_coupling_layer" : 8,
    "units_per_coupling_layer" : 128,
    "activation" : "relu", # seems to work much better than elu
    "dropout" : True,
    "summary_dimensions" : 1,
    "epochs" : 15,
    "batch_size" : 2**8,
    "early_stopping" : True,
    "learning_rate" : 1e-3,
    "coupling_design" : "spline",
    "permutation" : "learnable",
    "decay_rate" : 0.5,  
    }
  latent_dim = 2 # This seems broken for latent_dim=1 when using spline
else:
  hyperparameters = {
    "num_coupling_layer" : 8,
    "units_per_coupling_layer" : 128,
    "activation" : "relu",
    "dropout" : True,
    "summary_dimensions" : 1,
    "epochs" : 3,
    "batch_size" : 2**9,
    "early_stopping" : True,
    "learning_rate" : 1e-3,
    "coupling_design" : "affine",
    "permutation" : "learnable",
    "decay_rate" : 0.5,  
    }
  latent_dim = 1

### Data generation ###
print("- Making datasets")
cent = 172.5
diff = 7.5
ranges = [cent-diff,cent+diff]
array_size = (10**6)
sig_frac = 0.3

# Define context variables as the top mass
if not args.use_discrete:
  C = np.random.uniform(ranges[0], ranges[1], size=array_size)
else:
  C = np.array([])
  n_stochastic = 15
  for c in np.linspace(ranges[0]+0.5, ranges[1]-0.5, num=n_stochastic):
    C = np.concatenate((C,c*np.ones(int(round(array_size/n_stochastic)))))
  np.random.shuffle(C)
  C = C[:array_size]


# Define transformed variable X as a reconstructed mass
def MakeX(C, res=0.01):
  np.random.seed(seed_value)
  if not args.add_background:
    dist = np.random.normal(C, res*C)
  else:
    dist = np.empty(len(C))

    # signal events
    num_samples = int(round(len(C) * (sig_frac)))
    selected_indices = np.sort(np.random.choice(len(C), num_samples, replace=False))
    remaining_indices = np.setdiff1d(np.arange(len(C)), selected_indices)
    selected_C = C[selected_indices]
    dist[selected_indices] = np.random.normal(selected_C, res*selected_C)

    # background events
    lambda_param = 0.1
    const = 100.0
    bkg_ranges = [160.0,185.0]
    desired_num_samples = int(round((1-sig_frac) * len(C)))
    bkg_dist = np.empty(desired_num_samples)
    ind = 0
    while ind < desired_num_samples:
      # Generate exponential samples
      rand_uniform = np.random.uniform(0, 1, desired_num_samples * 2)  # Generate more samples than needed
      exponential_samples = -1 / lambda_param * np.log(1 - rand_uniform) + const

      # Filter and select samples within the range
      mask = (exponential_samples > bkg_ranges[0]) & (exponential_samples < bkg_ranges[1])
      valid_samples = exponential_samples[mask]

      remaining_space = desired_num_samples - ind
      num_valid_samples = min(remaining_space, valid_samples.shape[0])

      bkg_dist[ind:ind + num_valid_samples] = valid_samples[:num_valid_samples]
      ind += num_valid_samples

    # Resize the array if necessary
    dist[remaining_indices] = bkg_dist[:desired_num_samples]
  return dist

X = MakeX(C)

def Norm(x, mean=None, std=None):
  if mean == None or std == None:
    return ((x-x.mean())/x.std()).astype(np.float32), x.mean(), x.std()
  else:
    return ((x-mean)/std).astype(np.float32)

X, X_mean, X_std = Norm(X)
C, C_mean, C_std = Norm(C)

# Set up input in the format bayesflow takes it in
train = {}
test = {}

train["prior_draws"] = X[:int(array_size/2)].reshape(int(array_size/2),1)
test["prior_draws"] = X[int(array_size/2):].reshape(int(array_size/2),1)

if args.use_summary_network: # The extra dimension is needed to use a summary network
  train["sim_data"] = C[:int(array_size/2)].reshape(int(array_size/2),1,1)
  test["sim_data"] = C[int(array_size/2):].reshape(int(array_size/2),1,1)
else:
  train["sim_data"] = C[:int(array_size/2)].reshape(int(array_size/2),1)
  test["sim_data"] = C[int(array_size/2):].reshape(int(array_size/2),1)

# Set up models
print("- Setting up models")
inference_net = bf.networks.InvertibleNetwork(
  num_params=latent_dim,
  num_coupling_layers=hyperparameters["num_coupling_layer"],
  permutation=hyperparameters["permutation"],
  coupling_design=hyperparameters["coupling_design"],
  coupling_settings={
    "dense_args": dict(
      kernel_regularizer=None, 
      units=hyperparameters["units_per_coupling_layer"], 
      activation=hyperparameters["activation"]), 
    "dropout": hyperparameters["dropout"],
    },
  )

if args.use_summary_network:
  summary_net = bf.networks.DeepSet(summary_dim=hyperparameters["summary_dimensions"]) 
  amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
else:
  amortizer = bf.amortizers.AmortizedPosterior(inference_net)

# Define here because useful for a lot of the next steps
C_plot = np.linspace(170.0, 175.0, num=6, endpoint=True)

### Plot initial inputs ##
if not args.skip_initial_distribution:
  print("- Plotting initial inputs")

  nps = 100000
  nbins = 40

  for ind, tv in enumerate(C_plot): # Get binning
    if ind == 0:
      td = MakeX(tv*np.ones(nps))
    else:
      td = np.concatenate([td,MakeX(tv*np.ones(nps))])

  _, bins = np.histogram(td,bins=nbins)

  fig, ax = plt.subplots()
  hep.cms.text("Work in progress",ax=ax)
  for tv in C_plot:
    plt.plot(bins[:-1], np.histogram(MakeX(tv*np.ones(nps)),bins=bins)[0], label="y="+str(tv))
  plt.xlabel("x")
  plt.ylabel('Events')
  plt.legend()
  plt.tight_layout()
  plt.savefig("plots/{}_initial_distributions.pdf".format(name))
  print("Created plots/{}_initial_distributions.pdf".format(name))

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

  # Train model
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    trainer.default_lr, 
    decay_steps=int(len(train["prior_draws"])/hyperparameters["batch_size"]), 
    decay_rate=hyperparameters["decay_rate"]
  )
  history = trainer.train_offline(
    train, 
    epochs=hyperparameters["epochs"], 
    batch_size=hyperparameters["batch_size"], 
    validation_sims=test, 
    early_stopping=hyperparameters["early_stopping"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
  )
  
  # Plot the loss
  f = bf.diagnostics.plot_losses(history["train_losses"], history["val_losses"])
  f.savefig("plots/{}_loss.pdf".format(name))


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


### Test closure by using it as a generator ###
if not args.skip_generation:
  print("- Plotting generated distributions")

  # Parameters for plotting
  n_samples = 100000
  nbins = 40

  for val in C_plot:

    # Set up input dictionary
    inf_data = {}
    if args.use_summary_network:
      inf_data["summary_conditions"] = Norm(val*np.ones(n_samples).reshape((n_samples,1,1)), mean=C_mean, std=C_std)
    else:
      inf_data["direct_conditions"] = Norm(val*np.ones(n_samples).reshape((n_samples,1)), mean=C_mean, std=C_std)

    # Make synthetic dataset
    synth = (amortizer.sample(inf_data, 1)*X_std) + X_mean

    # Get the simulated dataset
    sim = MakeX(val*np.ones(n_samples))

    # Make the histograms
    sim_hist, bins = np.histogram(sim,bins=nbins)
    synth_hist, _ = np.histogram(synth,bins=bins)
    
    # Plot synthetic dataset against the simulated dataset
    plot_histogram_with_ratio(
      sim_hist, 
      synth_hist, 
      bins, 
      name_1='Simulated', 
      name_2='Synthetic',
      xlabel="x",
      name="plots/{}_synthetic_{}".format(name,str(val).replace(".","p")), 
      title_right = "y = {}".format(val),
      density = True,
      use_stat_err = True,
      )


### Need to integrate the pdf and ensure the integral is 1 ###

# For every true value we need to integrate and renormalise
def GetNormProb(test_value, data, integral_range=[140.0,200.0], integral_bins=10000, log=False):
  # Get Integral
  x_int = np.linspace(integral_range[0],integral_range[1],num=integral_bins)
  inf_data = {}
  inf_data["parameters"] = Norm(x_int.reshape((len(x_int),1)), mean=X_mean, std=X_std)
  if args.use_summary_network:
    inf_data["summary_conditions"] = Norm(test_value*np.ones(len(x_int)).reshape((len(x_int),1,1)), mean=C_mean, std=C_std)
  else:
    inf_data["direct_conditions"] = Norm(test_value*np.ones(len(x_int)).reshape((len(x_int),1)), mean=C_mean, std=C_std)    
  prob = np.exp(amortizer.log_posterior(inf_data))
  bin_width = x_int[1] - x_int[0]
  integral = np.sum(prob * bin_width)

  # Get unnormalised probability 
  inf_data = {}
  inf_data["parameters"] = Norm(data.reshape((len(data),1)), mean=X_mean, std=X_std)
  if args.use_summary_network:
    inf_data["summary_conditions"] = Norm(test_value*np.ones(len(data)).reshape((len(data),1,1)), mean=C_mean, std=C_std)
  else:
    inf_data["direct_conditions"] = Norm(test_value*np.ones(len(data)).reshape((len(data),1)), mean=C_mean, std=C_std)
  prob = np.exp(amortizer.log_posterior(inf_data))

  # Normalise probability
  prob = prob/integral

  if log:
    return np.log(prob)
  else:
    return prob
  

### Make a plot of the probabilities of a true value when varying the data ###
if not args.skip_probability:
  x_plot = np.linspace(150.0,190.0,num=100)
  for true_value in C_plot: 
    prob = GetNormProb(true_value, x_plot)
    fig, ax = plt.subplots()
    hep.cms.text("Work in progress",ax=ax)
    plt.plot(x_plot, prob, linestyle='-')
    plt.xlabel("x")
    plt.ylabel('p(x|{})'.format(true_value))
    plt.tight_layout()
    plt.savefig("plots/{}_probability_{}.pdf".format(name,str(true_value).replace(".","p")))
    print("Created plots/{}_probability_{}.pdf".format(name,str(true_value).replace(".","p")))


### Get best fit and draw likelihoods ###
if not args.skip_inference:
  print("- Getting the best fit value and drawing likelihood scans")

  # Number of events for scan
  test_array_size = int(round(1000/sig_frac))

  # Inputs for minimisation
  initial_guess = 175.0
  par_ranges = [160,180]

  # Function to get -2deltaLL from a tested context value for a given true value
  def objective_function(c, true_value, shift=0, absolute=False, par_range=None):
    if par_range != None:
      if not (c > par_range[0] and c < par_range[1]):
        return np.inf
    true_data = MakeX(true_value*np.ones(test_array_size))
    true_data = true_data.reshape((len(true_data),1))
    mtnll = -2*GetNormProb(c, true_data, log=True).sum()
    mtnll -= shift
    if absolute: mtnll = abs(mtnll)
    return mtnll

  for true_value in C_plot:

    # Get best fit
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', args=(true_value,0,False,par_ranges))

    # Getting crossings
    crossings = {
      0 : result.x[0]
    }
  
    for sigma in [-3,-2,-1,1,2,3]:

      # Set ranges to make sure we are getting the up or down crossing
      if sigma < 0:
        pr = [par_ranges[0],result.x]
      else:
        pr = [result.x,par_ranges[1]]

      # Get crossings
      crossings[sigma] = minimize(objective_function, initial_guess, method='Nelder-Mead', args=(true_value,result.fun+(sigma**2),True,pr)).x[0]

    # Calculate likelihood points
    n_points = 40
    if not args.extend_likelihood_range:
      plot_masses = np.linspace(crossings[-3],crossings[3],num=n_points+1, endpoint=True)
    else:
      plot_masses = np.linspace(cent-diff,cent+diff,num=n_points+1, endpoint=True)
    plot_m2dlls = [objective_function(c,true_value,shift=result.fun) for c in plot_masses]

    # Draw likelihood
    plot_likelihood(
      plot_masses, 
      plot_m2dlls, 
      crossings, 
      name="plots/{}_likelihood_{}".format(name,str(true_value).replace(".","p")), 
      xlabel="y", 
      true_value=true_value,
      cap_at_3=(not args.extend_likelihood_range)
      )

    # Print values
    dp = 3
    if crossings[0] > true_value:
      approx_deviation = (crossings[0] - true_value)/(crossings[0]-crossings[-1])
    else:
      approx_deviation = (true_value - crossings[0])/(crossings[1]-crossings[0])
    print("True Value: {}, Calculated Value: {} + {} - {}, Approx. deviation: {}".format(true_value,round(crossings[0],dp),round(crossings[1]-crossings[0],dp),round(crossings[0]-crossings[-1],dp),round(approx_deviation,dp)))


    # Make plot of initial distributions with the attempted inferred distribution on, as well as generated distribution at the mass points - this should help us see if the inference is working
    nps = 100000
    nbins = 40
    check_plot = [true_value,round(crossings[0],2)]
    colors = ['red', 'blue', 'green', 'purple']

    for ind, tv in enumerate(check_plot): # Get binning
      if ind == 0:
        td = MakeX(tv*np.ones(nps))
      else:
        td = np.concatenate([td,MakeX(tv*np.ones(nps))])

    _, bins = np.histogram(td,bins=nbins)

    fig, ax = plt.subplots()
    hep.cms.text("Work in progress",ax=ax)
    for ind, tv in enumerate(check_plot):
      hist = np.histogram(MakeX(tv*np.ones(nps)),bins=bins)[0]
      plt.plot(bins[:-1], hist/sum(hist), label="y="+str(tv), linestyle='-', color=colors[ind])

    for ind, val in enumerate(check_plot):
      inf_data = {}
      if args.use_summary_network:
        inf_data["summary_conditions"] = Norm(val*np.ones(nps).reshape((nps,1,1)), mean=C_mean, std=C_std)
      else:
        inf_data["direct_conditions"] = Norm(val*np.ones(nps).reshape((nps,1)), mean=C_mean, std=C_std)
      synth = (amortizer.sample(inf_data, 1)*X_std) + X_mean
      hist = np.histogram(synth,bins=bins)[0]
      plt.plot(bins[:-1], hist/sum(hist), label="gen y="+str(val), linestyle='--', color=colors[ind])

    inf_dist = MakeX(true_value*np.ones(test_array_size)).reshape((test_array_size,1))
    inf_hist = np.histogram(inf_dist, bins=bins)[0]
    inf_hist_err = np.sqrt(inf_hist)
    plt.errorbar(bins[:-1], inf_hist/sum(inf_hist), yerr=inf_hist_err/sum(inf_hist), label="data y={}".format(true_value), markerfacecolor='none', linestyle='None', fmt='k+')
    
    plt.xlabel("x")
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/{}_inferred_distributions_{}.pdf".format(name,str(true_value).replace(".","p")))
    print("Created plots/{}_inferred_distributions_{}.pdf".format(name,str(true_value).replace(".","p")))
