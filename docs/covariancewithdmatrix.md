---
layout: page
title: "Step: CovarianceWithDMatrix"
---

Compute the sandwich covariance using the Hessian and score matrix.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="CovarianceWithDMatrix"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `freeze_ind`, `val_ind`, `nuisance`, `variation`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** InitialFit best-fit result, Hessian and DMatrix outputs.

**Produces:** covariancewithdmatrix_{validation_index}.yaml and per-parameter covariancewithdmatrix_results files.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/CovarianceWithDMatrix{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

See [Inference options](inferenceoptions.md) for data selection, likelihood types, frozen full-fit values and optional evaluation caches. Reuse identical data, model and freeze settings across fitting and uncertainty stages.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |

## Behaviour and checks

The sandwich covariance is `H^{-1} D H^{-1}`. This addresses weighted-sample uncertainty estimation; it does not correct a biased learned density or a misspecified model.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/infer.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
