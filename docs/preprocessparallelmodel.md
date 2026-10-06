---
layout: page
title: "Step: PreProcessParallelModel"
---

Build and transform training/testing datasets independently for each model type and parameter.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PreProcessParallelModel"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `model_type`, `parameter_name`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Base train/test split tables and parameters_initial.yaml.

**Produces:** Final density/regression/classifier parquet tables and parameters_model_type_*.yaml fragments.

| Location | Path pattern |
| --- | --- |
| Results | `$PREP_DATA_DIR/$CFG_NAME/PreProcess/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

See [PreProcess](preprocess.md) for the data transformations and [parallel preprocessing](preprocess.md#parallel-preprocessing) for phase ordering.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--number-of-shuffles` | `20` | The number of times to loop through the dataset when shuffling in preprocess |

## Behaviour and checks

A classifier-specific job can be selected with `--specific="file_name=ttbar;category=run2;model_type=classifier_models;parameter_name=AbsoluteScale"`. Density jobs use `model_type=density_models`. Numbered parquet shards are temporary writer output, not complete training tables. A failed job may leave an old parameter fragment, so verify the final parquet files before starting training.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/preprocess.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
