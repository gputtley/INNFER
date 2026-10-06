---
layout: page
title: "Step: Scan"
---

Profile the likelihood at each selected value of a scanned parameter.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="Scan"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `column`, `freeze_ind`, `scan_ind`, `val_ind`, `nuisance`, `variation`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** ScanPoints grid, InitialFit result unless skip-initial-fit is used, and matching likelihood/data inputs.

**Produces:** Scan{suffix}/{process}/scan_values_{column}_{validation_index}_{scan_index}.yaml.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/Scan{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

See [Inference options](inferenceoptions.md) for data selection, likelihood types, frozen full-fit values and optional evaluation caches. Reuse identical data, model and freeze settings across fitting and uncertainty stages.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |
| `--number-of-scan-points` | `41` | The number of scan points run |
| `--sigma-between-scan-points` | `0.2` | The estimated sigma between the scanning points |
| `--skip-initial-fit` | `False` | Skip the initial fit if running a scan |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/infer.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
