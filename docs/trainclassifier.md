---
layout: page
title: "Step: TrainClassifier"
---

Train a likelihood-ratio classifier between varied and reference simulation.

The architecture accepts optional `log_l2_lambda`: when set, the L2 coefficient
is `10 ** log_l2_lambda` and takes precedence over `l2_lambda`. For example,
`log_l2_lambda: -6.0` selects `1e-6`. Leave it `null` to retain the linear
coefficient. This controls explicit L2, independently of AdamW weight decay.

For small nuisance responses, the FCNN architecture can set
`full_dataset_gradient: true` (default `false`). Each epoch then makes one
optimizer update using the complete signed training population. `batch_size`
sets the chunk size used to accumulate that gradient. Training and Test arrays
are cached in host memory; only a chunk's activations occupy GPU memory.
Weights are normalized once over the full training population, retaining
negative weights and contributions from chunks with zero signed weight.
Explicit L2, clipping and AdamW decay are applied once per update. Learning-rate
schedules count one update per epoch. This mode requires
`normalise_batch_categories: false` and uses full float32 model arithmetic
with float64 loss reductions and gradient accumulation.

Set `epochs: null` to specify `number_of_updates` instead. This must be a
positive integer. Training rounds **up to complete epochs**: ordinary
minibatch training uses
`ceil(number_of_updates / ceil(training_rows / batch_size))` epochs;
full-dataset training uses `number_of_updates` epochs because it makes one
update per epoch. For example, 2000 requested updates with 139 batches per
epoch becomes 15 epochs and 2085 updates. A non-null `epochs` takes precedence
and ignores `number_of_updates`. Early stopping can still end training before
the resolved budget; disable it for a fixed-length comparison.

When `epochs` is null and `full_dataset_gradient` is true,
`full_dataset_number_of_updates` optionally overrides `number_of_updates`.
It defaults to null, retaining the common budget if no override is supplied.
Minibatch mode ignores this override; a non-null `epochs` ignores both update
budgets. Separate budgets allow full-dataset trials to use fewer, more expensive
updates than minibatch trials.

The saved model architecture contains the resolved integer `epochs` alongside
the requested `number_of_updates`, so existing epoch-based tools can read it.
The input architecture is unchanged. Bayesian tuning accepts boolean
categorical choices, so an update-budget comparison can use:

```yaml
epochs: null
number_of_updates: 60000
full_dataset_number_of_updates: 2000
full_dataset_gradient: [false, true]
normalise_batch_categories: false
early_stopping: false
```

The default classifier architecture and classifier scan architectures use
`epochs: null`, `number_of_updates: 60000`, and
`full_dataset_number_of_updates: 2000`. The minibatch budget rounds the
approximately 58830 updates used to train the balanced-split 2223
AbsoluteMPFBias classifier; full-dataset mode uses the 2000-update diagnostic
budget instead. Early stopping remains configured separately.

`loss_evaluation_interval` defaults to `1`; in full-dataset mode it can be
increased to evaluate Train/Test losses less often. Epochs 0, 1 and the final
epoch are always evaluated. The main weights file contains the best evaluated
Test checkpoint, as in standard training; `--save-model-per-epoch` additionally
saves epoch checkpoints. The FCNN architecture's `checkpoint_interval` defaults
to `1` (every epoch); set it to `100`, for example, to save every 100 epochs.
The initial weights and final scheduled epoch are also saved when the flag is
enabled. In full-dataset mode each epoch is one update. These files are
independent of the best-Test main weights file. Early-stopping patience counts
loss evaluations.

`configs/architecture/classifier_btm_cferr_full_gradient.yaml` reproduces the
continuous-input cferr diagnostic's 2000-update setup using all configured
observables and the nuisance condition.

`configs/architecture/classifier_btm_cferr_full_gradient_4000.yaml` repeats that
study for 4000 updates, saving every 100 updates when used with
`--save-model-per-epoch`. It retains fixed checkpoints even if the best global
Test loss occurs earlier.

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
