---
layout: page
title: "Step: HyperparameterScanCollect"
---

Choose the best grid-scan candidate according to the requested metric.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="HyperparameterScanCollect"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** HyperparameterScan trial metrics, weights and architectures.

**Produces:** The selected nominal model .h5 and _architecture.yaml under the model directory.

| Location | Path pattern |
| --- | --- |
| Results | `$MODELS_DIR/$CFG_NAME/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--hyperparameter-metric` | `'loss_test,min'` | Comma separated metric name and whether you want max or min, separated by a comma. |
| `--model-type` | `'density'` | The model type to run the step for, if applicable. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/hyperparameter_scan_collect.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
