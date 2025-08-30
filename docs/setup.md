---
layout: page
title: "Setup"
---

# Setup Guide

This guide will help you install and configure INNFER.

- [Installing Repository and Packages](#installing-repository-and-packages)
- [Source Environment](#source-environment)
- [Setup SnakeMake](#setup-snakemake)

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

## Source Environment

At the beginning of every session you will need to run the following command to start the environment.
```bash
source env.sh
```

## Setup SnakeMake

The framework is set up to work with the SnakeMake workflow manager. Firstly, this needs to be setup for the local batch service. To do this for HTCondor run through the following steps:

```bash
source setup.sh snakemake_condor
```

You should call the profile `htcondor` and if you wish to looks at the condor submission logs then set this directory to somewhere accessible.

Next: [Building the Configuration](config.md).