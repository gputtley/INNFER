---
layout: page
title: "Step: SummaryPerVal"
---

Plot parameter summaries for one validation hypothesis at a time.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="SummaryPerVal"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Fit/uncertainty summaries for the selected validation indices.

**Produces:** summary_per_val_{index}_page1.pdf and additional pages as produced by the runner.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/SummaryPerVal{summary_from}Plot{extra_output_dir_name}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

`--summary-from` selects the upstream result family (for example Scan, Covariance or CovarianceWithDMatrix). Keep its directory suffix and validation indexing consistent with the producing steps.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |
| `--other-input` | `None` | Other inputs to likelihood and summary plotting |
| `--summary-nominal-name` | `'Nominal'` | Name of nominal summary points |
| `--summary-show-2sigma` | `False` | Show 2 sigma band on the summary. |
| `--use-scenario-labels` | `False` | Use Scenario 1, for example, labelling on plots rather than the string name |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/summary_per_val.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
