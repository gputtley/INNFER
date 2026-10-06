---
layout: page
title: "Step: BayesianHyperparameterTuning"
---

Search architecture/training parameters using Bayesian optimisation and optionally promote the best candidate.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="BayesianHyperparameterTuning"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Preprocessed model data and a Bayesian scan architecture; earlier timeout markers when continuing a timeout chain.

**Produces:** Trial metrics/architectures and results; optional nominal best-model files; timeout runs declare tuning_with_timeout_indices_ran_{index}.yaml.

| Location | Path pattern |
| --- | --- |
| Promoted model | `$MODELS_DIR/$CFG_NAME/{model_name}{extra_density_model_name}` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/BayesianHyperparameterTuning{extra_output_dir_name}/{model_name}{extra_density_model_name}` |
| Plots | `$PLOTS_DIR/$CFG_NAME/BayesianHyperparameterTuning{extra_output_dir_name}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--hyperparameter-metric` | `'loss_test,min'` | Comma separated metric name and whether you want max or min, separated by a comma. |
| `--load-weights-for-training` | `None` | Path to weights to load to start the training at |
| `--model-type` | `'density'` | The model type to run the step for, if applicable. |
| `--number-of-trials` | `10` | The number of trials to test for BayesianHyperparameterTuning |
| `--tuning-load-trials` | `None` | Comma separated list of trial indices to load or colon separated range |
| `--tuning-no-copy` | `False` | Do not copy the best model and architecture to the output directory |
| `--tuning-timeout-duration` | `9000` | Duration of the timeout for the Bayesian tuning |
| `--tuning-timeout-index` | `0` | Index of the timeout for the Bayesian tuning |
| `--tuning-use-timeout` | `False` | Using timeout for the Bayesian tuning |
| `--use-wandb` | `False` | Use wandb for logging. |
| `--wandb-project-name` | `'innfer'` | Name of project on wandb |

## Behaviour and checks

Choose `--model-type=density`, `classifier` or `regression`, with the corresponding architecture flag. Bayesian architecture ranges are interpreted by this runner; they are not the grid-scan format. `--tuning-no-copy` leaves trial results without promoting a nominal model. Timeout index zero starts the chain; each later index depends on the previous `tuning_with_timeout_indices_ran_*.yaml`. The final stage should promote the best model if downstream steps expect nominal weights.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/bayesian_hyperparameter_tuning.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
