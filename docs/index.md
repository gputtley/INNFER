---
layout: page
title: "Introduction"
permalink: /
---

# Welcome to INNFER

INNFER is a framework for ...

Running INNFER happens though the `scripts/innfer.py` script with an accompanying yaml config file parsed with the `--cfg` option (or a benchmark name with the `--benchmark` option). You also need to specify the step you want to run with the `--step` option. More information about these two options are detailed further below. An example command for this is shown below.
```bash
innfer --cfg="example_cfg.yaml" --step="PreProcess"
```

As some commands may take some time, jobs can be parallelised and submitted to a batch system such as HTCondor, by adding `--submit="condor.yaml"`, which points to a configuration file for submission to the batch. Running INNFER on a batch is highly recommended in all cases.

Example workflows are available in the workflows folder.

👉 Get started by reading the [Setup Guide](setup.md).