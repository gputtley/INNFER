---
layout: page
title: "Step: PlotDensity"
---

Compare density samples with model training/testing distributions, including conditional slices.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PlotDensity"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** EvaluateDensity synthetic tables and matching PreProcess tables.

**Produces:** comparison_{column}_{split}_{slice}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/PlotDensity/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/plot_density.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
