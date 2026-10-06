---
layout: page
title: "Step: MakePostFitUncertaintyAsimov"
---

Generate post-fit variations for distribution uncertainty bands.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="MakePostFitUncertaintyAsimov"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `asimov_file_name`, `val_ind`, `category`, `nuisance`, `nuisance_value`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Best-fit and uncertainty results, trained models and metadata.

**Produces:** Shifted asimov.parquet files under MakePostFitUncertaintyAsimov.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/MakePostFitUncertaintyAsimov{extra_output_dir_name}/{process}/{asimov_file_name}/{category}/val_ind_{asimov_val_ind}/{nuisance}/{nuisance_value}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

`--summary-from` selects the upstream result family (for example Scan, Covariance or CovarianceWithDMatrix). Keep its directory suffix and validation indexing consistent with the producing steps.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--use-spline` | `False` | Opt in to saved classifier/regression normalisation splines. |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--only-density` | `False` | Build asimov from only the density model |
| `--prefit-nuisance-constraints` | `False` | Make postfit plots with prefit nuisance constraints |
| `--prefit-nuisance-values` | `False` | Make postfit plots with prefit nuisance values |
| `--prune-classifier-models` | `None` | Comma separated list of key>values keep shape effects for |
| `--prune-from` | `'EvaluateClassifier'` | Step to prune from |
| `--use-asimov-scaling` | `10` | Generate asimov with this scaling up of the predicted yield |

With `--classifier-divide-by-nominal`, classifiers whose resolved evaluation parameter is zero are automatically skipped, including their model, spline and pruning dependencies.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/make_asimov.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
