---
layout: page
title: "Step: InputPlotTraining"
---

Inspect the physical and transformed distributions of model training and testing datasets.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="InputPlotTraining"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** PreProcess model tables and their parameters.yaml metadata.

**Produces:** distributions_{column}_{split}.pdf and transformed/unrolled variants in InputPlotTraining/{model_name}.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/InputPlotTraining{extra_output_dir_name}/{model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--plot-2d-unrolled` | `False` | Make 2D unrolled plots when running generator. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/input_plot_training.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
