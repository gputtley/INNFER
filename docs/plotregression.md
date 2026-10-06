---
layout: page
title: "Step: PlotRegression"
---

Compare binned averages of target weight variations with regression predictions.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PlotRegression"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** EvaluateRegression predictions and matching PreProcess regression tables.

**Produces:** average_weight_{column}_{split}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/PlotRegression/{model_name}{extra_regression_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/plot_regression.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
