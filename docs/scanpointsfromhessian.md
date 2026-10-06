---
layout: page
title: "Step: ScanPointsFromHessian"
---

Choose a profile-scan grid using the inverse-Hessian uncertainty.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ScanPointsFromHessian"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `column`, `freeze_ind`, `val_ind`, `nuisance`, `variation`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** InitialFit result and completed Hessian.

**Produces:** ScanPoints{suffix}/{process}/scan_ranges_{column}_{validation_index}.yaml.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/ScanPoints{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

See [Inference options](inferenceoptions.md) for data selection, likelihood types, frozen full-fit values and optional evaluation caches. Reuse identical data, model and freeze settings across fitting and uncertainty stages.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |
| `--number-of-scan-points` | `41` | The number of scan points run |
| `--scan-points-input` | `None` | Input for scan points, comma separated bracketed list of min and max, e.g. (171.5,173.5) |
| `--sigma-between-scan-points` | `0.2` | The estimated sigma between the scanning points |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/infer.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
