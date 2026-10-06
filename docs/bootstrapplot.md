---
layout: page
title: "Step: BootstrapPlot"
---

Plot distributions of bootstrap fitted parameters.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="BootstrapPlot"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `freeze_ind`, `column`, `val_ind`, `nuisance`, `variation`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Collected bootstrap result files.

**Produces:** bootstrap_distribution_{column}_{validation_index}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/BootstrapPlot{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/bootstrap_plot.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
