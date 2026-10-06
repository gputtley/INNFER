---
layout: page
title: "Step: ImpactsPlot"
---

Plot nuisance constraints and impacts on the selected target parameter.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ImpactsPlot"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `freeze_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** ImpactsCollect or ApproximateImpacts results and the selected fit/uncertainty summary.

**Produces:** impacts_page{page}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/ImpactsPlot{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

`--summary-from` selects the upstream result family (for example Scan, Covariance or CovarianceWithDMatrix). Keep its directory suffix and validation indexing consistent with the producing steps.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--impacts-from` | `'ApproximateImpacts'` | The source of the impacts to use for plotting (ApproximateImpacts, Impacts) |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/impacts_plot.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
