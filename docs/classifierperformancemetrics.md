---
layout: page
title: "Step: ClassifierPerformanceMetrics"
---

Compute the requested loss and distribution metrics for trained classifiers.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ClassifierPerformanceMetrics"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`, `extra_name`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Trained classifiers and the preprocessed model/validation data needed by the selected metrics.

**Produces:** metrics.yaml under ClassifierPerformanceMetrics/{model_name}, with optional diagnostic plots.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/ClassifierPerformanceMetrics{extra_output_dir_name}/{model_name}{extra_classifier_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--classifier-performance-metrics` | `'loss,histogram,multidim,chi_squared,kl_divergence'` | Comma separated list of classifier performance metrics |
| `--loop-over-epochs` | `False` | Loop over epochs for performance metrics |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/classifier_performance_metrics.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
