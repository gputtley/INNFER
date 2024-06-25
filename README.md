# INNFER - Invertible Neural Networks for Extracting Results

## Installing Repository and Packages

To import the github repository, clone with the following command.
```
git clone https://github.com/gputtley/INNFER.git
```

Then to set up the conda environment for running this repository run this command. You will need to click `enter` and `yes` through all prompts. For the conda installation license hit `q` to go the end of the file and then you can fit `enter`. 
```
source setup.sh conda
```
```
source setup.sh packages
```

## Setup Environment

At the beginning of every session you will need to run the following command to setup the environment.
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

The input configuration file is how you give key information to INNFER about the input datasets and their preprocessing, how you want to validate the models and how you want to build the likelihood. An example of this is shown below, for a stat-only top mass measurement.

```
name: TopMass
files:
  other: data/top_mass_other.parquet  
  ttbar: data/top_mass_ttbar.parquet
variables:
  - top_mass_rec
  - W_mass_rec
  - b_pt_rec
pois:
  - top_mass_true
nuisances: []
preprocess:
  selection: "(jec==0)"
  standardise: all
  train_test_val_split: 0.6:0.1:0.3
  equalise_y_wts: True
  train_test_y_vals:
    mass:
    - 166.5
    - 169.5
    - 171.5
    - 173.5
    - 175.5
    - 178.5
  validation_y_vals:
    mass:
    - 171.5
    - 172.5
    - 173.5    
inference:
  rate_parameters:
  - ttbar
  nuisance_constraints: {}
validation: 
  rate_parameter_vals:
    ttbar:
    - 0.8
    - 1.0
    - 1.2
data_file: data/top_mass_data.parquet
```

The `PreProcess` step will preprocess all the simulated files and the data_file. It will apply the selections to the datasets, select the variables/pois/nuisances specified, standardise the columns of both simulated datasets, train/test/val split the dataset, equalise the weights in each Y category for training and give specific Y values for training/testing to validation. These numbers are set up like this as typically validation on the border Y values does not show good performance, so this range should be larger enough that the borders do not matter in the fit to data. The 172.5 validation sample also gives us a good measure of the interpolation ability of the models. We running the `Train`, separate models will be trained for the two simulated files, trained on the train split and tested on the test split. The inference option gives how the combined likelihood will be build. Included the parsing of rate parameters and nuisance constraints. Finally, the validation option in this case dictates which values of the rate parameter you want to validate for.

## Snakemake

The framework is set up to work with the SnakeMake workflow manager. Firstly, this needs to be setup for the local batch service. To do this for HTCondor run through the following steps:

```
source setup.sh snakemake_condor
```

You should call the profile `htcondor` and if you wish to looks at the condor submission logs then set this directory to somewhere accessible.

To use condor workflows you can set the required steps and submission options in the snakemake configuration file. An example file is `configs/snakemake/condor_core.yaml`. This contains the core steps of the innfer package. You can then run with snakemake by parsing `--step=SnakeMake --snakemake-cfg=condor_core.yaml`. Other snakemake submission options are not set up. Please contact us if you wish for this to be setup.

Snakemake workflows are defined by a yaml file detailing the steps to run, the options to parse for each step and the submission options. An example of this is in `configs/snakemake/condor_core_quick.yaml`. It is recommended that your run infer command to use snakemake in a `tmux` terminal, so your terminal cannot be disconnected. INNFER can then be run with the following command:

```
python3 scripts/innfer.py --cfg="boosted_top_mass.yaml" --step="SnakeMake" --snakemake-cfg="condor_core_quick.yaml"
```

## Running from a benchmark scenario

There are a number of benchmark scenarios setup in the `python/worker/benchmarks.py` class, that can be used to generate datasets (with a known density) and a running configuration file. These can be generated with the command:

```
python3 scripts/innfer.py --benchmark="GaussianWithExpBkg" --step="MakeBenchmark"
```

You can the continue to run the remaining steps with the `--benchmark="GaussianWithExpBkg"` option rather than using the `--cfg` option. You can also use SnakeMake for this whole process with the relevant snakemake configuration files.

## Description of Steps

`MakeBenchmark`: This will create a dataset and a yaml config file to run fromm for the benchmark scenario specified (with `--benchmark=GaussianWithExpBkg` for example) in the `python/worker/benchmarks.py` class.

`PreProcess`: This will take preprocess the input dataset ready for training, including perform standardisation and train/test/val splitting of the dataset. It will also produce a parameters file which contains crucial information about the created datasets that will be used in later steps.

`Train`: This will train invertible neural networks, using BayesFlow, to learn the probability density functions of the datasets provided. Specific network architectures can be specified with `--architecture="default.yaml"` for example.

`PerformanceMetrics`: This will calculate performance metrics on the trained networks and write them to a yaml file.

`HyperparameterScan`: This will perform a hyperparameter scan based on a scan architecture file given by `--architecture="scan.yaml"` for example. If you want to track your trainings with wandb, you first must need to set up an account at [https://wandb.ai/](https://wandb.ai/). Then you must connect your terminal to your account with the command `wandb login`. You will need to enter the API given on the website. Now you can add `--use-wandb` to the command and track online.

`HyperparameterScanCollect`: This will collect the best performing hyperparameter scan model and copy it and its architecture into the directories required to use that model for the next steps.

`Generator`: This will make plots using the network as a generator and comparing it to the input simulation. By default this will only make 1D plots for every Y value. However, you can make 2d plots with the `--plot-2d-unrolled` option.

`GeneratorSummary`: This will make a summary plot of the comparison between the simulation and using the network as a generator, showing all unique Y values.

## Structure of Code

The code is built on several classes and sets of functions in the `python` directory. These are split into two types `runner` and `worker`. The `runner` files are typically called by the `scripts/innfer.py` and each represent a specific running step. The `worker` files are called by the `runner` files in order to help perform the required steps. Here I will give an explanation to the `worker` classes as the `runner` files are fairly self explanatory and the motivation for them is described in the description of steps.

`batch.py`: Add stuff

`benchmarks.py`: Add stuff

`combined_network.py`: Add stuff

`data_loader.py`: Add stuff

`data_processor.py`: Add stuff

`innfer_trainer.py`: Add stuff

`likelihood.py`: Add stuff

`network.py`: Add stuff

`plotting.py`: Add stuff

`useful_functions.py`: Add stuff