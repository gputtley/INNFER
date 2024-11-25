# INNFER - Invertible Neural Networks for Extracting Results

## Installing Repository and Packages

To import the github repository, clone with the following command.
```
git clone https://github.com/gputtley/INNFER.git
```

Then to install conda run this command. You will need to click `enter` and `yes` through all prompts. If you already have a conda installed, this is not needed. 
```
source setup.sh conda
```

To set up the environment, you will need to run this command.
```
source setup.sh env
```

## Setup Environment

At the beginning of every session you will need to run the following command to start the environment.
```
source env.sh
```

## Running INNFER

Running INNFER happens though the `scripts/innfer.py` script with an accompanying yaml config file parsed with the `--cfg` option (or a benchmark name with the `--benchmark` option). You also need to specify the step you want to run with the `--step` option. More information about these two options are detailed further below. An example command for this is shown below.
```
python3 scripts/innfer.py --cfg="boosted_top_mass.yaml" --step="PreProcess"
```

As some commands may take some time, jobs can be parallelised and submitted to a batch system such as HTCondor, by adding `--submit="condor.yaml"`, which points to a configuration file for submission to the batch. Running INNFER on a batch is highly recommended in all cases.

## Input Configuration File

The input configuration file is how you give key information to INNFER about the input datasets and their preprocessing, how you want to validate the models and how you want to build the likelihood. Examples of this is shown in the `configs/run` directory.

The `PreProcess` step will preprocess all the simulated files and the data_file given in the yaml file. It will apply the selections to the datasets, select the variables/pois/nuisances specified, standardise the columns of both simulated datasets, train/test/val split the dataset, equalise the weights in each Y category for training and give specific Y values for training/testing to validation. When running the `Train` step, separate models will be trained for the two simulated files, trained on the train split and tested on the test split. The inference option gives how the combined likelihood will be build. Included the parsing of rate parameters and nuisance constraints. Finally, the validation option in this case dictates which values of the rate parameter you want to validate for.

## Snakemake

The framework is set up to work with the SnakeMake workflow manager. Firstly, this needs to be setup for the local batch service. To do this for HTCondor run through the following steps:

```
source setup.sh snakemake_condor
```

You should call the profile `htcondor` and if you wish to looks at the condor submission logs then set this directory to somewhere accessible.

To use condor workflows you can set the required steps and submission options in the snakemake configuration file. An example file is `configs/snakemake/condor_core.yaml`. This contains the core steps of the innfer package. You can then run with snakemake by parsing `--step=SnakeMake --snakemake-cfg=condor_core.yaml`. Other snakemake submission options are not set up. Please contact us if you wish for this to be setup.

Snakemake workflows are defined by a yaml file detailing the steps to run, the options to parse for each step and the submission options. Examples of this are in the `configs/snakemake` directory. It is recommended that your run infer command to use snakemake in a `tmux` terminal, so your terminal cannot be disconnected. 

As the PreProcess (and MakeBenchmark) step needs to be run before creating the jobs, this is typically run first (although can be added to the snakemake config without submission options):
```
python3 scripts/innfer.py --cfg="config.yaml" --step="PreProcess"
```

The snakemake chain can then be run with the following command:
```
python3 scripts/innfer.py --cfg="config.yaml" --step="SnakeMake" --snakemake-cfg="condor_core_quick.yaml"
```

## Running from a benchmark scenario

There are a number of benchmark scenarios setup in the `python/worker/benchmarks.py` class, that can be used to generate datasets (with a known density) and a running configuration file. These can be generated with the command:

```
python3 scripts/innfer.py --benchmark="Dim1GaussianWithExpBkg" --step="MakeBenchmark"
```

You can the continue to run the remaining steps with the `--benchmark="Dim1GaussianWithExpBkg"` option rather than using the `--cfg` option. You can also use SnakeMake for this whole process with the relevant snakemake configuration files.

## Description of Steps

`MakeBenchmark`: This will create a dataset and a yaml config file to run fromm for the benchmark scenario specified (with `--benchmark=Dim1GaussianWithExpBkg` for example) in the `python/worker/benchmarks.py` class.

`PreProcess`: This will take preprocess the input dataset ready for training, including perform standardisation and train/test/val splitting of the dataset. It will also produce a parameters file which contains crucial information about the created datasets that will be used in later steps.

`Custom`: This is a customisable step. You can place your own runner class in the `python/runner/custom_module` directory and call the module with the `--custom-module` option.

`InputPlot`: This will plot the datasets that are an input to the training and the validation steps.

`Train`: This will train invertible neural networks, using BayesFlow, to learn the probability density functions of the datasets provided. Specific network architectures can be specified with `--architecture="default.yaml"` for example.

`PerformanceMetrics`: This will calculate performance metrics on the trained networks and write them to a yaml file.

`HyperparameterScan`: This will perform a hyperparameter scan based on a scan architecture file given by `--architecture="scan.yaml"` for example. If you want to track your trainings with wandb, you first must need to set up an account at [https://wandb.ai/](https://wandb.ai/). Then you must connect your terminal to your account with the command `wandb login`. You will need to enter the API given on the website. Now you can add `--use-wandb` to the command and track online.

`HyperparameterScanCollect`: This will collect the best performing hyperparameter scan model and copy it and its architecture into the directories required to use that model for the next steps.

`BayesianHyperparameterTuning`: This will perform a bayesian hyperparameter tuning based on the ranges given in by `--architecture=scan_bayesian.yaml` for example. This can also be used with wandb.

`SplitValidationFiles`: This will split the validation dataset into separate files for each unique Y values. This will speed up infer steps, if called with the `--split-validation-files` option.

`Flow`: This will plot the distributions of the normalising flow as you move through the coupling layers.

`Generator`: This will make plots using the network as a generator and comparing it to the input simulation. By default this will only make 1D plots for every Y value. However, you can make 2d plots with the `--plot-2d-unrolled` option.

`GeneratorSummary`: This will make a summary plot of the comparison between the simulation and using the network as a generator, showing all unique Y values.

`MakeAsimov`: This will create the asimov dataset. The number of events generated can be changed with the `-n-asimov-events` option.

`LikelihoodDebug`: This is a way to debug likelihood problems. It allows you to evaluate the likelihood with the parameter values given by the `--other-input` option.

`InitialFit`: This will perform a minimisation of the negative log likelihood. Whenever calling the infer module you can specify the `--likelihood-type`, `--data-type` and numerous other options. For the minimisation you can also specifiy the `--minimisation-method`.

`ApproximateUncertainty`: This will calculate approximate uncertainties by shifting from the minimum of the likelihood and interpolating to the relevant crossings.

`Hessian`: This will calculate the Hessian matrix.

`Covariance`: This will calculate the covariance matrix from the inverse of the Hessian, as well as the uncertainties on the parameters.

`DMatrix`: This will determine the d-matrix, as derived [here](https://arxiv.org/abs/1911.01303).

`CovarianceWithDMatrix`: This will determine the asymptotically correct covariance matrix, as well as the uncertainties on the parameters, from the Hessian and DMatrix calculations.

`ScanPoints`: This will find sensible points to perform a likelihood scan for. The number of scan points can be given by the `--number-of-scan-points` option.

`Scan`: This will perform a profiled likelihood scan.

`ScanCollect`: This will collect the profiled likelihood scan.

`ScanPlot`: This will plot the profiled likelihood scan.

`BootstrapInitialFits`: This will bootstrap the resampling of the dataset and produce negative log likelihood minimisations for every bootstrapped dataset. The number of bootstraps can be controlled by the `--number_of_bootstraps` option.

`BootstrapCollect`: This will collect the bootstrapped results into a single file.

`BootstrapPlot`: This will plot the distribution of bootstrapped results.

`BestFitDistributions`: This will draw the distributions of the best fit results found in the fit versus the data used for the fit.

`SummaryChiSquared`: This will calculate the chi squared value between the best fit and the truth for all validation parameters.

`Summary`: This wil plot a summary of the best fit and truth values found in validation.