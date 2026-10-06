---
layout: page
title: "Step: MakeAsimovForNormalisationSpline"
---

Generate density-only integration samples for classifier normalisation.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="MakeAsimovForNormalisationSpline"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** The trained density model and its preprocessing metadata.

**Produces:** MakeAsimovForNormalisationSpline/{density_model_name}/asimov.parquet.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/MakeAsimovForNormalisationSpline{extra_output_dir_name}/{model_name}` |
| Models read | `$MODELS_DIR/$CFG_NAME` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--classifier-divide-by-nominal` | `False` | Divide the classifier by the nominal value |
| `--use-spline` | `False` | Opt in to saved classifier/regression normalisation splines. |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |

With `--classifier-divide-by-nominal`, classifiers whose resolved evaluation parameter is zero are automatically skipped, including their model, spline and pruning dependencies.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/make_asimov.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
