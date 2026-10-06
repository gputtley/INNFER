---
layout: page
title: "Step: MakePostFitAsimov"
---

Generate synthetic predictions at the fitted parameter values.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="MakePostFitAsimov"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `asimov_file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** InitialFit best-fit files, trained density/ratio models and preprocessing metadata.

**Produces:** asimov.parquet files under MakePostFitAsimov for each selected process/category/validation point.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/MakePostFitAsimov{extra_output_dir_name}/{process}/{asimov_file_name}/{category}/val_ind_{asimov_val_ind}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--no-spline` | `False` | Do not use the normalisaing splines when creating asimov |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--only-density` | `False` | Build asimov from only the density model |
| `--prefit-nuisance-values` | `False` | Make postfit plots with prefit nuisance values |
| `--prune-classifier-models` | `None` | Comma separated list of key>values keep shape effects for |
| `--prune-from` | `'EvaluateClassifier'` | Step to prune from |
| `--use-asimov-scaling` | `10` | Generate asimov with this scaling up of the predicted yield |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/make_asimov.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
