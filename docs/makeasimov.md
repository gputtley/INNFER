---
layout: page
title: "Step: MakeAsimov"
---

Generate finite weighted synthetic samples at configured validation hypotheses.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="MakeAsimov"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Trained density and applicable ratio models, metadata, and any normalisation splines or pruning metrics requested.

**Produces:** MakeAsimov/{process}/{category}/val_ind_{index}/asimov.parquet.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/MakeAsimov{extra_output_dir_name}/{process}/{category}/val_ind_{val_ind}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--classifier-divide-by-nominal` | `False` | Divide the classifier by the nominal value |
| `--no-spline` | `False` | Do not use the normalisaing splines when creating asimov |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--only-default-asimov` | `False` | Build asimov for only the default validation indices |
| `--only-density` | `False` | Build asimov from only the density model |
| `--prune-classifier-models` | `None` | Comma separated list of key>values keep shape effects for |
| `--prune-from` | `'EvaluateClassifier'` | Step to prune from |
| `--use-asimov-scaling` | `10` | Generate asimov with this scaling up of the predicted yield |

## Behaviour and checks

These are finite Monte Carlo samples with weights, rather than noiseless event-level datasets. The workflow passes yield-based `--use-asimov-scaling` (default 10); the runner uses its fixed event count only when `use_asimov_scaling` is None. The CLI scaling option is integer-valued and takes precedence over the fixed-count option in this dispatch. `--only-density` omits learned nuisance-ratio corrections; `--no-spline` disables their saved normalisation splines. Keep these choices consistent with the validation/inference target.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/make_asimov.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
