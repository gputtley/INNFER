---
layout: page
title: "Step: ScanCollect"
---

Collect scan-point likelihood values and derive the fitted result and crossing uncertainties.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ScanCollect"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `column`, `freeze_ind`, `val_ind`, `nuisance`, `variation`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** All requested Scan point files.

**Produces:** scan_results_{column}_{validation_index}.yaml under ScanCollect.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/ScanCollect{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |
| `--number-of-scan-points` | `41` | The number of scan points run |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/scan_collect.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
