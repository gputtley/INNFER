---
layout: page
title: "Step: MakeAsimovNuisanceVariations"
---

Generate synthetic up/down nuisance variations at the nominal validation hypothesis.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="MakeAsimovNuisanceVariations"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `nuisance`, `shift`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** The same model inputs as MakeAsimov, including the selected nuisance models.

**Produces:** MakeAsimovNuisanceVariations/{process}/{category}/{nuisance}_{up_or_down}/asimov.parquet.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/MakeAsimovNuisanceVariations{extra_output_dir_name}/{process}/{category}/{nuisance}_{shift}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--classifier-divide-by-nominal` | `False` | Divide the classifier by the nominal value |
| `--use-spline` | `False` | Opt in to saved classifier/regression normalisation splines. |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--only-density` | `False` | Build asimov from only the density model |
| `--prune-classifier-models` | `None` | Comma separated list of key>values keep shape effects for |
| `--prune-from` | `'EvaluateClassifier'` | Step to prune from |
| `--use-asimov-scaling` | `10` | Generate asimov with this scaling up of the predicted yield |

With `--classifier-divide-by-nominal`, classifiers whose resolved evaluation parameter is zero are automatically skipped, including their model, spline and pruning dependencies.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/make_asimov.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
