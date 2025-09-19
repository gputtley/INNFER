---
layout: page
title: "Setup"
---

## Cloning from github

The INNFER repository can be directly cloned from github with the following command

```bash
git clone https://github.com/gputtley/INNFER.git
```

## Installing conda and the environment

Running on the code relies on multiple python packages which are installed using a conda environment. Firstly, conda needs to set up. To install this you can run the following command. You may need to click `enter` and `yes` through all prompts. If you already have conda installed, this initial command may not be needed. 
```bash
source setup.sh conda
```

Next, you will need to set up the environment, with the following command.
```bash
source setup.sh env
```

## Source environment

At the beginning of every session you will need to run the following command to start the environment. This will initiate the conda environment and set a number of useful global variables.
```bash
source env.sh
```

## Setup SnakeMake

The framework is set up to work with the SnakeMake workflow manager. Firstly, this needs to be setup for the local batch service. To do this for HTCondor run through the following steps.
```bash
source setup.sh snakemake_condor
```
You should call the profile `htcondor` and if you wish to looks at the condor submission logs then set this directory to somewhere accessible. Note, this is only setup for use on HTCondor batch systems.

<br>

---

Next: [Running INNFER](running.md).