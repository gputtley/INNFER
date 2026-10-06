---
layout: page
title: "Step: Summary"
---

Plot fitted values and uncertainties against validation truth across hypotheses.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="Summary"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Results selected by summary-from; optional comparison and chi-squared summaries.

**Produces:** summary.pdf under Summary{suffix}/{process}.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/Summary{summary_from}Plot{extra_output_dir_name}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

`--summary-from` selects the upstream result family (for example Scan, Covariance or CovarianceWithDMatrix). Keep its directory suffix and validation indexing consistent with the producing steps.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--add-specific-category-to-dir-name` | `False` | Add the specific category name to the directory name |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |
| `--other-input` | `None` | Other inputs to likelihood and summary plotting |
| `--summary-nominal-name` | `'Nominal'` | Name of nominal summary points |
| `--summary-show-2sigma` | `False` | Show 2 sigma band on the summary. |
| `--summary-show-chi-squared` | `False` | Add the chi squared value to the plot |
| `--summary-subtract` | `False` | Use subtraction instead of division in summary |
| `--use-scenario-labels` | `False` | Use Scenario 1, for example, labelling on plots rather than the string name |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/summary.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
