---
layout: page
title: "Step: PreProcessParallelValidation"
---

Build physical validation tables for one hypothesis per job.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PreProcessParallelValidation"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `val_ind`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Base split tables, initial metadata and validation hypotheses.

**Produces:** val_ind_{index}/X_{split}.parquet, Y_{split}.parquet, wt_{split}.parquet and validation parameter fragments.

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

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/preprocess.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
