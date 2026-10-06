---
layout: page
title: "Step: ResampleValidationForData"
---

Create a mock observed dataset by resampling the default simulation validation samples.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ResampleValidationForData"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Default validation datasets for the selected processes/categories.

**Produces:** DataCategories/{category}/data.parquet. This occupies the same location as categorised real data.

| Location | Path pattern |
| --- | --- |
| Results | `$PREP_DATA_DIR/$CFG_NAME/DataCategories/{category}/data.parquet` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Behaviour and checks

This uses the same data-category path as DataCategories and returns without replacing it if that output already exists. The resampling path drops negative weights. Use a separate preparation directory when preserving observed data alongside mock data. Resampled validation events are for checking the data workflow, not an independent closure sample.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/resample_validation_for_data.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
