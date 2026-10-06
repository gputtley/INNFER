---
layout: page
title: "Step: SummaryAllButOneCollect"
---

Collect fits in which all parameters except one were frozen into a common summary input.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="SummaryAllButOneCollect"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Per-parameter all-but-one fit/uncertainty results.

**Produces:** Collected result YAML files under SummaryAllButOneCollect.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/{summary_from}{extra_output_dir_name}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

`--summary-from` selects the upstream result family (for example Scan, Covariance or CovarianceWithDMatrix). Keep its directory suffix and validation indexing consistent with the producing steps.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/summary_all_but_one_collect.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
