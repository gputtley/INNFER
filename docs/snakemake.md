---
layout: page
title: "SnakeMake"
---

# SnakeMake

To use condor workflows you can set the required steps and submission options in the snakemake configuration file. Example files are in the `configs/snakemake` directory. This contains the core steps of the innfer package. You can then run with snakemake by parsing `--step="SnakeMake" --snakemake-cfg="example_snakemake_cfg.yaml"`. Other snakemake submission options are not set up. Please contact us if you wish for this to be setup.

Snakemake workflows are defined by a yaml file detailing the steps to run, the options to parse for each step and the submission options. Examples of this are in the `configs/snakemake` directory. It is recommended that your run infer command to use snakemake in a `tmux` terminal, so your terminal cannot be disconnected. 

The snakemake chain can then be run with the following command:
```bash
innfer --cfg="example_cfg.yaml" --step="SnakeMake" --snakemake-cfg="example_snakemake_cfg.yaml"
```