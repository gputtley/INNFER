---
layout: page
title: "Step: InputPlotComparingTrainWithVariations"
---

Compare model training distributions with nominal and shifted inference datasets.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="InputPlotComparingTrainWithVariations"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `nuisance`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** PreProcess training tables, nominal validation tables and nuisance variations.

**Produces:** train_vs_variations_{column}_{nuisance}.pdf and double-ratio plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/InputPlotComparingTrainWithVariations{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--ratio-range` | `'0.5,1.5'` | Range for ratio plot |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/input_plot_comparing_train_with_variations.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
