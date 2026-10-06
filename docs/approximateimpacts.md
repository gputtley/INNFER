---
layout: page
title: "Step: ApproximateImpacts"
---

Estimate parameter impacts from covariance correlations without additional fits.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ApproximateImpacts"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `freeze_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** The Covariance output selected by the input-directory suffix and the configured target parameter.

**Produces:** An impacts YAML summary under ApproximateImpacts.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/ApproximateImpacts{extra_output_dir_name}{freeze_suffix}/{process}/impacts_{val_ind}.yaml` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

This dispatch reads Covariance directly; it does not switch covariance families with --summary-from. Match the input-directory suffix and validation indexing to that producer.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--impact-to` | `None` | Variable to get impact to |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/approximate_impacts.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
