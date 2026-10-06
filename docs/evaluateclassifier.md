---
layout: page
title: "Step: EvaluateClassifier"
---

Evaluate classifier predictions and summarise their performance for inspection or pruning.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="EvaluateClassifier"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Trained classifier, architecture, metadata and classifier training/testing tables.

**Produces:** pred_train.parquet, pred_test.parquet and performance_metrics.yaml.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/EvaluateClassifier/{model_name}{extra_classifier_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/evaluate_classifier.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
