---
layout: page
title: "Step: DataCategories"
---

Apply observed-data calculations, selections, category cuts and configured dequantisation.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="DataCategories"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** data_file and data_selection in the run configuration; optional data_add_columns and data_calculate.

**Produces:** DataCategories/{category}/data.parquet and, for configured binned fits, data_binned_fit.yaml.

| Location | Path pattern |
| --- | --- |
| Results | `$PREP_DATA_DIR/$CFG_NAME/DataCategories/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Observed-data representation

`data_add_columns` adds per-file metadata, and `data_calculate` evaluates configured derived features. The runner then applies the data/category selection and the shared `preprocess.dequantisation` transform before writing data.parquet. This keeps quantised observables in the same physical representation as the model's simulation datasets. Model standardisation is applied later during density evaluation, rather than stored in the observed table.

Dequantisation uses uniform noise inside floating-point rounding cells, with the configured physical bounds. See [PreProcess](preprocess.md#dequantisation) for the accepted configuration and input-grid checks. Rebuild DataCategories after changing these settings.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/data_categories.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
