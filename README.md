# INNFER - Invertible Neural Networks for Extracting Results

## TO DO

`Setup`: Currently there is a bayesflow/snakemake package conflict. We are not actually using bayesflow package imported at the moment but this should be fixed.
`Setup`: It would be good to import the packages required from a yaml file install of pip installing all of them.
`PreProcess`: Implement discrete to continuous code
`Train/HyperparameterScan`: Implement wandb
`Collect`: Write a collection worker class, this will work for all collect steps in the innfer.
`Yield`: Make a yield function file, you should be able to interchange this for something custom.
`CombinedModel`: Write class to combine models together and incorporate rate parameters.
`Generator`: Write generator class.
`Likelihood`: Write likelihood class.
`Bootstrap`: Write bootstrap class.
`Infer`: Write infer class.
`Summary`: Write a class to summarise the validation of inference and the asimov.
`Snakemake`: Write snakemake class.
`Snakemake`: Write instructions for snakemake installation.
`MakeParquetFromRoot`: Add the file to make parquet files from root.
`Testing`: Test it works with a nuisance.
`Other`: Docstrings and sort class functions.
`Other`: Finish README documentation.

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

Add stuff

## Description of Steps

`MakeBenchmark`: This will create a dataset and a yaml config file to run fromm for the benchmark scenario specified (with `--benchmark=GaussianWithExpBkg` for example) in the `python/worker/benchmarks.py` class.

`PreProcess`: This will take preprocess the input dataset ready for training, including perform standardisation and train/test/val splitting of the dataset. It will also produce a parameters file which contains crucial information about the created datasets that will be used in later steps.

`Train`: This will train invertible neural networks, using BayesFlow, to learn the probability density functions of the datasets provided. Specific network architectures can be specified with `--architecture="default.yaml"` for example.

`PerformanceMetrics`: This will calculate performance metrics on the trained networks and write them to a yaml file.

`HyperparameterScan`: This will perform a hyperparameter scan based on a scan architecture file given by `--architecture="scan.yaml"` for example. If you want to track your trainings with wandb, you first must need to set up an account at [https://wandb.ai/](https://wandb.ai/). Then you must connect your terminal to your account with the command `wandb login`. You will need to enter the API given on the website. Now you can add `--use-wandb` to the command and track online.

`HyperparameterScanCollect`: This will collect the best performing hyperparameter scan model and copy it and its architecture into the directories required to use that model for the next steps.

## Snakemake

The framework is set up to work with the SnakeMake workflow manager. Firstly, this needs to be setup for the local batch service. To do this for HTCondor run through the following steps:

```
INSERT SNAKEMAKE CONDOR SETUP STEPS
```
To use condor workflows you can set the required steps and submission options in the snakemake configuration file. An example file is `configs/snakemake/condor_core.yaml`. This contains the core steps of the innfer package. You can then run with snakemake by parsing `--step=SnakeMake --snakemake-cfg=condor_core.yaml`.

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