<img src="images/Imperial_College_London_logo.png" alt="Imperial College Logo" width="200">

<hr>

# INNFER - Invertible Neural Networks for Extracting Results

## Table of Contents
- [Installing Repository and Packages](#installing-repository-and-packages)
- [Setup Environment](#setup-environment)
- [Running INNFER](#running-innfer)
- [Input Configuration File](#input-configuration-file)
- [SnakeMake](#snakemake)
- [Description of Steps](#description-of-steps)

## Installing Repository and Packages

To import the github repository, clone with the following command.
```bash
git clone https://github.com/gputtley/INNFER.git
```

Then to install conda run this command. You will need to click `enter` and `yes` through all prompts. If you already have a conda installed, this is not needed. 
```bash
source setup.sh conda
```

To set up the environment, you will need to run this command.
```bash
source setup.sh env
```

## Setup Environment

At the beginning of every session you will need to run the following command to start the environment.
```bash
source env.sh
```

## Running INNFER

Running INNFER happens though the `scripts/innfer.py` script with an accompanying yaml config file parsed with the `--cfg` option (or a benchmark name with the `--benchmark` option). You also need to specify the step you want to run with the `--step` option. More information about these two options are detailed further below. An example command for this is shown below.
```bash
innfer --cfg="example_cfg.yaml" --step="PreProcess"
```

As some commands may take some time, jobs can be parallelised and submitted to a batch system such as HTCondor, by adding `--submit="condor.yaml"`, which points to a configuration file for submission to the batch. Running INNFER on a batch is highly recommended in all cases.

Example workflows are available in the workflows folder.

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

## SnakeMake

The framework is set up to work with the SnakeMake workflow manager. Firstly, this needs to be setup for the local batch service. To do this for HTCondor run through the following steps:

```bash
source setup.sh snakemake_condor
```

You should call the profile `htcondor` and if you wish to looks at the condor submission logs then set this directory to somewhere accessible.

To use condor workflows you can set the required steps and submission options in the snakemake configuration file. Example files are in the `configs/snakemake` directory. This contains the core steps of the innfer package. You can then run with snakemake by parsing `--step="SnakeMake" --snakemake-cfg="example_snakemake_cfg.yaml"`. Other snakemake submission options are not set up. Please contact us if you wish for this to be setup.

Snakemake workflows are defined by a yaml file detailing the steps to run, the options to parse for each step and the submission options. Examples of this are in the `configs/snakemake` directory. It is recommended that your run infer command to use snakemake in a `tmux` terminal, so your terminal cannot be disconnected. 

The snakemake chain can then be run with the following command:
```bash
innfer --cfg="example_cfg.yaml" --step="SnakeMake" --snakemake-cfg="example_snakemake_cfg.yaml"
```

## Description of Steps

`MakeBenchmark` : This will create a dataset and a yaml config file to run fromm for the benchmark scenario specified (with `--benchmark="Dim5"` for example) in the `python/worker/benchmarks.py` class.

`LoadData`: Loads data to create base datasets.

`PreProcess` : Prepares datasets for training, testing, and validation. Includes standardisation, train/test/validation splitting, and optional binned fit input.

`Custom` : Runs a custom module specified by the `--custom-module` option. Additional options can be passed using `--custom-options`.

`InputPlotTraining` : Plots training and testing datasets after preprocessing.

`InputPlotValidation` : Plots validation datasets after preprocessing.

`TrainDensity` : Trains density networks using the specified architecture. Supports logging with wandb and saving models per epoch.

`SetupDensityFromBenchmark` : Sets up density models based on a benchmark scenario.

`TrainRegression` : Trains regression networks for specific parameters using the specified architecture.

`EvaluateRegression` : Evaluates regression models on validation datasets, also makes the normalisation spline.

`PlotRegression` : Plots regression results of average binned value vs average regressed value.

`MakeAsimov` : Generates Asimov datasets for validation. The number of events can be specified using `--number-of-asimov-events` or you can use the `--use-asimov-scaling` option to take the number of asimov events scaled from the predicted number of events.

`DensityPerformanceMetrics` : Computes performance metrics for trained density networks. Supports metrics like loss, histograms, and multidimensional metrics.

`EpochPerformanceMetricsPlot` : Plots performance metrics as a function of training epochs.

`PValueSimVsSynth` : Performs a p-value dataset comparison test between simulated and synthetic datasets.

`PValueSynthVsSynth` : Performs a p-value dataset comparison test between bootstrapped synthetic datasets.

`PValueSynthVsSynthCollect` : Collects results from the synthetic vs. synthetic p-value tests.

`PValueDatasetComparisonPlot` : Plots p-value dataset comparison results.

`HyperparameterScan` : This will perform a hyperparameter scan based on a scan architecture file given by `--architecture="scan.yaml"` for example. If you want to track your trainings with wandb, you first must need to set up an account at [https://wandb.ai/](https://wandb.ai/). Then you must connect your terminal to your account with the command `wandb login`. You will need to enter the API given on the website. Now you can add `--use-wandb` to the command and track online.

`HyperparameterScanCollect` : Collects the best-performing model from a hyperparameter scan.

`BayesianHyperparameterTuning` : Performs Bayesian hyperparameter tuning using a specified architecture. Supports logging with wandb in the same way `HyperparameterScan` does.

`Flow` : Visualizes the flow of the network through coupling layers.

`Generator` : Uses the network as a generator to create plots comparing network outputs to input simulations. Supports 1D and 2D unrolled plots.

`GeneratorSummary` : Creates summary plots comparing simulations and network-generated outputs across all unique Y values.

`LikelihoodDebug` : Debugs likelihood calculations for specific parameter values. It allows you to evaluate the likelihood with the parameter values given by the `--other-input` option.

`InitialFit` : Performs minimization of the negative log-likelihood to find the best fit.  Whenever calling the infer module you can specify the `--likelihood-type`, `--data-type` and numerous other options. Supports specifying the minimization method with `--minimisation-method`.

`ApproximateUncertainty` : Calculates approximate uncertainties by interpolating likelihood crossings.

`Hessian` : Computes the Hessian matrix for the likelihood.

`HessianParallel` : Computes the Hessian matrix in parallel for efficiency.

`HessianCollect` : Collects results from parallel Hessian computations.

`HessianNumerical` : Computes the Hessian matrix numerically.

`Covariance` : Calculates the covariance matrix from the inverse of the Hessian.

`DMatrix` : Computes the D-matrix for asymptotic corrections, as derived [here](https://arxiv.org/abs/1911.01303).

`CovarianceWithDMatrix` : Computes the corrected covariance matrix using the Hessian and D-matrix.

`ScanPointsFromApproximate and ScanPointsFromHessian` : Identifies points for likelihood scans based on approximate uncertainties or the Hessian matrix. The number of scan points can be given by the `--number-of-scan-points` option.

`Scan` : Performs a profiled likelihood scan over identified points.

`ScanCollect` : Collects results from likelihood scans.

`ScanPlot` : Plots results from likelihood scans.

`MakePostFitAsimov` : Generates post-fit Asimov datasets using best-fit parameter values.

`PostFitPlot` : Plots distributions for best-fit results compared to the data used for fitting.

`SummaryChiSquared` : Computes the chi-squared value between best-fit and truth values for validation parameters.

`SummaryAllButOneCollect` : Collects results for "all-but-one" parameter summaries.

`Summary` : Plots a summary of best-fit and truth values for validation parameters.

`SummaryPerVal` : Plots summaries of results for each validation index.