---
layout: page
title: "Step: TrainRegression"
---

Train a regression model for a configured parameter-dependent weight variation.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="TrainRegression"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Regression training/testing features, targets and weights, metadata and the regression architecture.

**Produces:** {process}.h5 and {process}_architecture.yaml under regression_{process}_{parameter}_{category}.

| Location | Path pattern |
| --- | --- |
| Results | `$MODELS_DIR/$CFG_NAME/{model_name}{extra_regression_model_name}` |
| Plots | `$PLOTS_DIR/$CFG_NAME/TrainRegression/{model_name}{extra_regression_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--regression-architecture` | `'configs/architecture/regression_default.yaml'` | Architecture for regression model |
| `--save-model-per-epoch` | `False` | Save a model at each epoch |
| `--use-wandb` | `False` | Use wandb for logging. |
| `--wandb-project-name` | `'innfer'` | Name of project on wandb |

## Behaviour and checks

Regression targets encode weight variations, rather than directly modelling an observable density. Run EvaluateRegression afterwards to build the normalisation spline required by downstream likelihood/generation configurations.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/train_regression.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
