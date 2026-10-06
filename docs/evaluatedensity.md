---
layout: page
title: "Step: EvaluateDensity"
---

Sample the trained density at the conditions in its training/testing tables.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="EvaluateDensity"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** A trained density model, saved architecture and preprocessing metadata/tables.

**Produces:** synth_train.parquet and synth_test.parquet under EvaluateDensity/{model_name}.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/EvaluateDensity/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/evaluate_density.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
