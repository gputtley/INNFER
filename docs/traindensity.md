---
layout: page
title: "Step: TrainDensity"
---

Train a conditional or unconditional density model using the selected architecture.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="TrainDensity"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Density X_train/Y_train/wt_train and X_test/Y_test/wt_test tables, parameters.yaml and the density architecture; optional starting weights.

**Produces:** {process}.h5 and {process}_architecture.yaml under the density model directory; optional epoch checkpoints and loss/LR plots.

| Location | Path pattern |
| --- | --- |
| Results | `$MODELS_DIR/$CFG_NAME/{model_name}{extra_density_model_name}` |
| Plots | `$PLOTS_DIR/$CFG_NAME/TrainDensity/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--density-architecture` | `'configs/architecture/density_default.yaml'` | Architecture for density model |
| `--load-weights-for-training` | `None` | Path to weights to load to start the training at |
| `--save-model-per-epoch` | `False` | Save a model at each epoch |
| `--train-from-nominal` | `False` | Train from the training dataset with the parameters at the nominal value |
| `--use-wandb` | `False` | Use wandb for logging. |
| `--wandb-project-name` | `'innfer'` | Name of project on wandb |

## Behaviour and checks

See [Density architecture](densityarchitecture.md) for coupling layers, summary networks, optimisers and buffered shuffling. Training saves the lowest finite validation-loss checkpoint, including the starting model at epoch zero. With `--load-weights-for-training`, keep the architecture compatible with the checkpoint; architecture changes generally cannot reuse weights directly. The accompanying preprocessing transforms must also remain compatible.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/train_density.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
