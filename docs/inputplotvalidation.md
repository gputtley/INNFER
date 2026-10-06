---
layout: page
title: "Step: InputPlotValidation"
---

Inspect physical simulation distributions at validation hypotheses and optionally compare simulation splits.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="InputPlotValidation"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** PreProcess validation tables and metadata.

**Produces:** X_distributions_{column}.pdf, optional split comparisons and weight-distribution PDFs.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/InputPlotValidation{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--compare-sim-types` | `False` | Compare different simulation types in InputPlotValidation |
| `--likelihood-type` | `'unbinned_extended'` | Type of likelihood to use for fitting. |
| `--plot-weight-distribution` | `False` | Plot weight distribution when running InputPlotValidation. |
| `--ratio-range` | `'0.5,1.5'` | Range for ratio plot |
| `--use-scenario-labels` | `False` | Use Scenario 1, for example, labelling on plots rather than the string name |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/input_plot_validation.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
