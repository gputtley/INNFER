---
layout: page
title: "Step: TrainClassifier"
---

Train a likelihood-ratio classifier between varied and reference simulation.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="TrainClassifier"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Classifier X_train/y_train/wt_train and X_test/y_test/wt_test tables, metadata and the classifier architecture.

**Produces:** {process}.h5 and {process}_architecture.yaml under classifier_{process}_{parameter}_{category}.

| Location | Path pattern |
| --- | --- |
| Results | `$MODELS_DIR/$CFG_NAME/{model_name}{extra_classifier_model_name}` |
| Plots | `$PLOTS_DIR/$CFG_NAME/TrainClassifier/{model_name}{extra_classifier_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--classifier-architecture` | `'configs/architecture/classifier_default.yaml'` | Architecture for classifier model |
| `--save-model-per-epoch` | `False` | Save a model at each epoch |
| `--use-wandb` | `False` | Use wandb for logging. |
| `--wandb-project-name` | `'innfer'` | Name of project on wandb |

## Behaviour and checks

The target column is lowercase `y`; density conditions use uppercase `Y`. Class balancing is performed during preprocessing. `--change-classifier-type-from-parameter` selects specialised interpolators for named parameters. Changing an architecture while reusing weights requires compatible tensor shapes.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/train_classifier.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
