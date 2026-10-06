---
layout: page
title: "Step: LoadData"
---

Read the configured input files and build the base simulation tables used by preprocessing.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="LoadData"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `base_file_name`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Run configuration, files entries, and their ROOT or parquet inputs.

**Produces:** One {base_file_name}.parquet per base dataset in LoadData.

| Location | Path pattern |
| --- | --- |
| Results | `$PREP_DATA_DIR/$CFG_NAME/LoadData{extra_output_dir_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Base-file processing

Each `files` entry defines its input paths, selection, metadata columns and optional `pre_calculate` functions. LoadData gathers the columns needed by model variations, category cuts, weights and configured extra columns, applies base calculations and writes numeric base tables. The downstream PreProcess step recalculates `calculate` quantities after applying parameter/feature shifts.

Base datasets can be shared by multiple model definitions. Use `--specific="base_file_name=your_base_file"` to select one LoadData iteration; `--specific-file-name` selects model processes in other steps and does not filter this base-file loop.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/load_data.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
