---
layout: page
title: "Step: EvaluateRegression"
---

Evaluate weight regressions and derive their normalisation splines.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="EvaluateRegression"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Trained regression weights/architecture and matching PreProcess regression tables.

**Produces:** pred_train.parquet, pred_test.parquet and {process}_norm_spline.pkl beside the model.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/EvaluateRegression/{model_name}{extra_regression_model_name}` |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/EvaluateRegression/{model_name}{extra_regression_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/evaluate_regression.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
