---
layout: page
title: "Step: ValidationPerformanceMetrics"
---

Compare complete synthetic predictions with simulation at validation points or nuisance variations.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ValidationPerformanceMetrics"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `val_dataset_ind`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** PreProcess simulation tables and corresponding MakeAsimov or nuisance-variation synthetic tables.

**Produces:** metrics.yaml in a separate directory for each selected validation dataset.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--skip-weight-variation` | `False` | For double variations skip the weight variation |
| `--validation-performance-metrics-datasets` | `'validation'` | Comma separated list of types of dataset (validation, nuisance_variations, nuisance_double_variations) |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/validation_performance_metrics.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
