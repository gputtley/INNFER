import numpy as np
import copy

def Minimise(
    function,
    gradient_function,
    initial_guess,
    options = {
      "max_iterations" : 100,
      "minimisation_threshold" : 0.01,
      "minimisation_threshold_number" : 5,
      "nan_gradient_scale" : 0.5,
      "gradient_step" : 0.001
    }
  ):

  # Set up stores
  values = []

  # Get initial guess value
  guess = copy.deepcopy(initial_guess)
  value = function(guess)
  gradients = gradient_function(guess)

  # Raise exception if nan
  if np.isnan(value):
    raise ValueError("Initial guess gave back a nan.")
  if np.isnan(np.array(gradients)).any():
    raise ValueError("Initial guess gradients gave back a nan.")


  # Begin iterations
  for iteration in range(options["max_iterations"]):

    # Compute updated value
    updated_guess = [guess[ind]-(options["gradient_step"]*gradients[ind]) for ind in range(len(guess))]
    updated_value = function(updated_guess)
    updated_gradients = gradient_function(updated_guess)

    # While loop for nan value
    nan_ind = 1
    while np.isnan(updated_value) or np.isnan(np.array(gradients)).any():
      updated_guess = [guess[ind]-((options["nan_gradient_scale"]**nan_ind)*gradients[ind]) for ind in range(len(guess))]
      updated_value = function(updated_guess)
      updated_gradients = gradient_function(updated_guess)
      nan_ind += 1

    # Save to stores
    values.append(value)

    # Set updated guess
    guess = copy.deepcopy(updated_guess)
    value = copy.deepcopy(updated_value)
    gradients = copy.deepcopy(updated_gradients)


    # Check finished
    if len(values) >= options["minimisation_threshold_number"]:
      diff = max(values) - min(values)
      if diff < options["minimisation_threshold"]:
        break

  return guess, value
