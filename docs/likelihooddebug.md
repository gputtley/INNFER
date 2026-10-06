---
layout: page
title: "Step: LikelihoodDebug"
---

Evaluate and print likelihood diagnostics at explicitly requested parameter points.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="LikelihoodDebug" --debug-input="bw_mass=172.5"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `nuisance`, `variation`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Likelihood model inputs and the selected simulation, synthetic or observed dataset.

**Produces:** Diagnostic terminal output; this step does not declare a fit-result artifact.

See [Inference options](inferenceoptions.md) for data selection, likelihood types, frozen full-fit values and optional evaluation caches. Reuse identical data, model and freeze settings across fitting and uncertainty stages.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--debug-input` | `None` | The conditions for the LikelihoodDebug step. This is semi colon separated, comma separated key=value inputs |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/infer.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
