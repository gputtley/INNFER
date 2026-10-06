---
layout: page
title: "Step: PlotClassifier"
---

Compare classifier-reweighted distributions with the shifted simulation target.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PlotClassifier"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** EvaluateClassifier predictions and the corresponding PreProcess tables.

**Produces:** reweighted_{column}_{split}_inclusive.pdf and conditional-bin plots.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Plots | `$PLOTS_DIR/$CFG_NAME/PlotClassifier/{model_name}{extra_classifier_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/plot_classifier.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
