---
layout: page
title: "Step: HyperparameterScan"
---

Train and evaluate a grid of candidate model architectures.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="HyperparameterScan"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`, `architecture_ind`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Preprocessed model data and a scan architecture for the selected model type.

**Produces:** Trial model/architecture and performance files under HyperparameterScan/{model_name}.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/HyperparameterScan{extra_output_dir_name}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--model-type` | `'density'` | The model type to run the step for, if applicable. |
| `--use-wandb` | `False` | Use wandb for logging. |
| `--wandb-project-name` | `'innfer'` | Name of project on wandb |

## Behaviour and checks

Use the architecture flag corresponding to `--model-type`. Run HyperparameterScanCollect after all candidates finish. The optimisation metric uses `metric,direction`, for example `loss_test,min`.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/hyperparameter_scan.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
