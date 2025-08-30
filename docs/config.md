---
layout: page
title: "Configuration File"
---

# Building the Configuration

Here’s how to define and customize your configuration...

## Input Configuration File

The input configuration file is how you give key information to INNFER about the input datasets and their preprocessing, how you want to validate the models and how you want to build the likelihood. Examples of this is shown in the `configs/run` directory.

The structure of the config should be as noted below.

```yaml
name: example_config # Name for all data, plot and model folders

variables: # Variables, parameters of interest (shape only - rates parameters added later) and nuisance parameters
  - var_1
  - var_2
pois:
  - poi_1
nuisances:
  - nui_1
  - nui_2
  - nui_3

data_file: data.root # Data file - this can be a root or a parquet file

inference: # Settings for inference
  nuisance_constraints: # Nuisance parameters to add Gaussian constraints
    - nui_1
    - nui_2
    - nui_3
  rate_parameters: # Processes to add rate parameters for
    - signal
  lnN: # Log normal uncertainties by name, rate (this can be asymmetric, parse as a [0.98,1.02] for example) and processes they effect
    nui_1: 
      rate: 1.02
      files: ["signal", "background"]

default_values: # Set default values - will set defaults of nuisances to 0 and rates to 1 if not set
  poi_1: 125.0

models: # Define information about the models needed - this is split by process
  signal:
    density_models: # This is the nominal density model that is trained. This is parsed as a list but typically only the first element is used, as each element of the list is varied separately
      - parameters: ["poi_1", "nui_2"] # Parameters in model
        file: base_signal # Base file to load from
        shifts: # Parameters to shift from base file
          nui_2:
            type: continuous # This can also be discrete or fixed
            range: [-3.0,3.0] # Range to vary parameter in
        n_copies: 10 # This is the number of copies to make of the base dataset
    regression_models: # These are regression models to vary the nominal density for weight variations. This is parsed as a list and each element of the list will be a new regression model
      - parameter: "nui_3"
        file: base_signal
        shifts:
          nui_3:
            type: continuous
            range: [-3.0,3.0]
        n_copies: 5
    yields: {"file": "base_signal"} # This is the base file to calculate the yield for, the default yield is done for the defined default parameters
  background:
    density_models:
      - parameters: ["nui_2"]
        file: base_background
        shifts:
          nui_2:
            type: continuous
            range: [-3.0,3.0]
        n_copies: 10
    regression_models: []
    yields: {"file": "base_background"}

validation: # This is the options for the validation files
  loop: # Loop of sets of parameters to validate. This is done for all processes individually and combined. For an individual file with a subset of the parameters, a unique subset of the loop is formed.
    - {"poi_1" : 125.0, "nui_1" : 0.0, "nui_2" : 0.0, "nui_3" : 0.0, "mu_signal" : 1.0}
    - {"poi_1" : 124.0, "nui_1" : 0.0, "nui_2" : 0.0, "nui_3" : 0.0, "mu_signal" : 1.0}
    - {"poi_1" : 126.0, "nui_1" : 0.0, "nui_2" : 0.0, "nui_3" : 0.0, "mu_signal" : 1.0}
    - {"poi_1" : 125.0, "nui_1" : -1.0, "nui_2" : 0.0, "nui_3" : 0.0, "mu_signal" : 1.0}
    - {"poi_1" : 125.0, "nui_1" : 1.0, "nui_2" : 0.0, "nui_3" : 0.0, "mu_signal" : 1.0}
    - {"poi_1" : 126.0, "nui_1" : 0.0, "nui_2" : -1.0, "nui_3" : 0.0, "mu_signal" : 1.0}
    - {"poi_1" : 126.0, "nui_1" : 0.0, "nui_2" : 1.0, "nui_3" : 0.0, "mu_signal" : 1.0}
    - {"poi_1" : 126.0, "nui_1" : 0.0, "nui_2" : 0.0, "nui_3" : -1.0, "mu_signal" : 1.0}
    - {"poi_1" : 126.0, "nui_1" : 0.0, "nui_2" : 0.0, "nui_3" : 1.0, "mu_signal" : 1.0}
    - {"poi_1" : 126.0, "nui_1" : 0.0, "nui_2" : 0.0, "nui_3" : 0.0, "mu_signal" : 0.8}
    - {"poi_1" : 126.0, "nui_1" : 0.0, "nui_2" : 0.0, "nui_3" : 0.0, "mu_signal" : 1.2}
  files: # Base files to generate the validation datasets from
    ttbar: base_signal
    other: base_background

preprocess: # Preprocess parameters
  train_test_val_split: 0.8:0.1:0.1 # How to train/test/val split the dataset
  save_extra_columns: {} # Extra columns to save during preprocessing - useful for custom modules

files: # Base dataset inputs
  base_signal: # Base dataset name
    inputs: # Input files
      - signal_123.root
      - signal_124.root
      - signal_125.root
      - signal_126.root
      - signal_127.root
    add_columns: # Add extra columns to dataset
      poi_1: 
        - 123.0
        - 124.0
        - 125.0
        - 126.0
        - 127.0
    tree_name: "tree" # Root ttree name
    selection: "(var_1>50)" # Initial selecton
    weight: "wt" # Name of event weights, scaled correctly
    parameters: ["poi_1"] # Parameters in the dataset before any shifts
    pre_calculate: # Extra variables to be calculated - including the ability to add feature morphing shifted variables
      var_2: var_2_uncorr * (1 + (nui_3 * sigma_3))
    post_calculate_selection: "(var_2 > 30)" # Selection after pre_calculate is calculated
    weight_shifts: # Weight shifts
      nui_3: "(1 + (nui_2 * sigma_2))"
  base_background:
    inputs:
      - background.root
    tree_name: "tree"
    selection: "(var_1>50)"
    weight: "wt"
    parameters: []
    pre_calculate: 
      var_2: var_2_uncorr * (1 + (nui_3 * sigma_3))
    post_calculate_selection: "(var_2 > 30)"
    weight_shifts:
      nui_3: "(1 + (nui_2 * sigma_2))"
```

---

Next: [SnakeMake](snakemake.md).