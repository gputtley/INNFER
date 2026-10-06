---
layout: page
title: "Step: PlotDensityTransform"
---

Inspect the fitted density preprocessing transforms and their gradients.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PlotDensityTransform"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Density transform metadata from PreProcess.

**Produces:** gaussian_transform_{column}.pdf, cdf_spline_{column}.pdf and gradient PDFs.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/PlotDensityTransform{extra_output_dir_name}/{model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/plot_density_transform.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
