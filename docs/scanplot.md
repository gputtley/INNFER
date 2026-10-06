---
layout: page
title: "Step: ScanPlot"
---

Plot collected profile-likelihood curves, optionally comparing fits or uncertainty breakdowns.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ScanPlot"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `column`, `freeze_ind`, `val_ind`, `nuisance`, `variation`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** ScanCollect results for the nominal fit and any other-input comparisons.

**Produces:** likelihood_scan_{column}_{validation_index}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/ScanPlot{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |
| `--other-input` | `None` | Other inputs to likelihood and summary plotting |
| `--rezero-scan` | `False` | Re-zero the likelihood scan, when running ScanPlot |
| `--scan-colours` | `None` | Comma separated list of colours to use for the scan points in the scan plot. |
| `--scan-linestyles` | `None` | Comma separated list of linestyles to use for the scan points in the scan plot. |
| `--scan-no-result-text` | `False` | Do not add the text to the plot when there is no result in the scan plot. |
| `--scan-nominal-name` | `'Nominal'` | Name of nominal for scan |
| `--scan-plot-breakdown` | `False` | Do the syst and stat breakdown, assuming the main input if the full and other-input is stat. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/scan_plot.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
