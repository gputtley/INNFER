---
layout: page
title: "Step: InputPlotNuisanceVariations"
---

Compare nominal simulation with the configured up/down nuisance variations.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="InputPlotNuisanceVariations"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `nuisance`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** PreProcess nominal and nuisance-variation tables; binned metadata when applicable.

**Produces:** nuisance_variation_{nuisance}_{column}_{sim_type}.pdf and optional binned variants.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/InputPlotNuisanceVariations{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--likelihood-type` | `'unbinned_extended'` | Type of likelihood to use for fitting. |
| `--ratio-range` | `'0.5,1.5'` | Range for ratio plot |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/input_plot_nuisance_variations.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
