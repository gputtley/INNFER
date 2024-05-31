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

**TO DO: Currently there is a bayesflow/snakemake package conflict. We are not actually using bayesflow at the moment, so this needs to be fixed.**

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
`HyperparameterScan`: This will perform a hyperparameter scan based on a scan architecture file given by `--architecture="scan.yaml"` for example.

## Snakemake

Add stuff