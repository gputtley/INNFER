---
layout: page
title: "Running INNFER"
---

Running INNFER happens though the `scripts/innfer.py` script, this can be called by the `innfer` alias. 

You must provide two accompanying options:
- A config file parsed with the `--cfg` option (discussed [here](config.md)), or a benchmark name with the `--benchmark` option.
- The step you want to run with the `--step` option. More information about the steps are detailed further below. If you want to run multiple steps in series, you can provide the required steps separated by a comma (`,`).

Therefore, an example command to run **INNFER** is shown below.
```bash
innfer --cfg="example_cfg.yaml" --step="PreProcess"
```

Each step is designed to iterate over all possible parallelisations for that step. For example, if you are training a separate density model for signal and background processes, then these are completely independent tasks that can be parallelised. You can submit each split to a batch system such as HTCondor, by adding `--submit="condor.yaml"`, which points to a file for submission to the batch. The submission options can be altered in this yaml file. If you want to run only an individual element of the loop, then you can do this by using the `--specific` option. For example, if you want to only run the signal density model loop, you would add `--specific="model_name=density_signal"`. If it is a nested loop, you can prove a semi-colon (`;`) separated list of the `key=value` to `--specific`. To find the keys and values, they are either printed to the terminal when running locally or you can look in the loop of the `scripts/innfer.py` script. You can also make use of this functionality when running locally. When submitting to the batch, if you want to group together some of the splits into a single job, you can use the `--points-per-job` option which by default is set to 1.

Example workflows that have been used in analyses are available in the workflows folder.

<br>

---

Next: [Steps](steps.md).