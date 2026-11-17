---
layout: page
title: "Density architecture"
---

[TrainDensity](traindensity.md) reads the YAML selected by `--density-architecture`. A copy is saved beside the chosen model weights. Run configurations define datasets and their transformations; architecture files define the model and training procedure.

## Flow and condition representation

| Key | Meaning |
| --- | --- |
| `type` | Density implementation; the default is `BayesFlow`. |
| `coupling_design` | `affine`, `spline` or `interleaved`. |
| `num_coupling_layers` | Number of coupling layers. |
| `permutation` | Feature-mixing permutation between coupling layers, such as `fixed` or `learnable`. |
| `affine_units_per_dense_layer`, `affine_num_dense_layers`, `affine_activation` | Affine conditioner size, depth and activation. |
| `spline_units_per_dense_layer`, `spline_num_dense_layers`, `spline_activation` | Spline conditioner size, depth and activation. |
| `spline_bins` | Number of spline bins. |
| `affine_dropout`, `spline_dropout` | Enable conditioner dropout; probabilities use the corresponding `_dropout_prob` keys. |
| `affine_mc_dropout`, `spline_mc_dropout` | Monte Carlo dropout options; avoid stochastic density evaluation when requiring a deterministic likelihood. |
| `use_summary_network` | Embed condition columns before passing them to the flow. This is a condition encoder, not a summary of the event dataset. |
| `summary_dim` | Output dimension of the condition encoder. |
| `summary_hidden_units`, `summary_activation` | Hidden layer widths and activation of the condition encoder. |

With `use_summary_network: False`, the flow sees the condition columns directly. Unconditional models use zero condition columns. Summary-network weights are saved in `{checkpoint_stem}_summary.h5` companion files by the network's weight-saving callback and must accompany a checkpoint that uses the encoder.

## Training

`epochs` and `batch_size` control passes and optimiser batch size. `learning_rate`, `optimizer_name`, `lr_scheduler_name` and `lr_scheduler_options` configure optimisation. `gradient_clipping_norm`, `l2_lambda` and dropout regularise training in the paths supporting them. Early-stopping settings are `early_stopping`, `patience`, `tolerance` and `wait_till`.

`log_l2_lambda` optionally overrides `l2_lambda` with `10 ** log_l2_lambda`.
For example, `log_l2_lambda: -6.0` gives an L2 coefficient of `1e-6`.
Leave it `null` to use `l2_lambda`, including `l2_lambda: 0` to disable explicit
L2. This setting does not change AdamW's separate optimizer weight decay.

For `CosineDecayWithWarmUp`, `warmup_fraction` sets the initial ramp, `step_power` changes the progress through the cosine, and this repository's `alpha` is an absolute terminal learning rate. `step_power: 1` gives ordinary cosine progress after warmup; a value below one decays faster early in that interval. Other schedulers have their own parameter conventions; see [optimizer.py](../python/worker/optimizer.py).

The trainer saves the checkpoint with the lowest finite evaluation loss, including epoch zero. Fine-tuning therefore preserves the loaded starting weights when no epoch improves the loss. `--save-model-per-epoch` also writes `_epoch_{index}.h5` checkpoints.

### Exponential moving average of weights

Both `BayesFlow` and `KerasFlow` support TensorFlow optimizer EMA:

```yaml
use_ema: true
ema_momentum: 0.999
```

`use_ema` defaults to `false`, including for older architectures without this
key. `ema_momentum` defaults to `0.999` and must be finite and in `[0, 1)` when
enabled. After each optimizer update the average becomes
`momentum * previous_average + (1 - momentum) * current_weights`.
The averaging window is measured in optimizer updates, not epochs; a momentum
of `0.999` corresponds to roughly 1,000 updates. A new TrainDensity/fine-tuning run
starts a fresh average from its initial/loaded weights.

When enabled, epoch Train/Test loss evaluation, early stopping, best-model
selection and ordinary checkpoint saving use EMA weights. The ordinary weights
are restored before further gradient updates. Epoch zero is the initial model.
This does not average losses or generated events, change dropout at inference,
or add evaluation cost to models loaded from the saved checkpoints.

The usual model directory contains EMA checkpoints. A sibling directory with
`_raw` appended contains the corresponding non-averaged weights and an
architecture YAML, including summary-network companion files where applicable.
For example:

```text
models/RUN/density_ttbar_24_study/ttbar.h5
models/RUN/density_ttbar_24_study/ttbar_epoch_1.h5
models/RUN/density_ttbar_24_study_raw/ttbar.h5
models/RUN/density_ttbar_24_study_raw/ttbar_epoch_1.h5
```

The raw main model is paired with the epoch selected by EMA Test loss; it is
not independently selected by raw Test loss. Raw epoch checkpoints are written
when `--save-model-per-epoch` is enabled. Step `Outputs()` includes both main
models and architecture files when EMA is enabled. For a model suffix `_study`,
use `--extra-density-model-name=_study` to evaluate EMA and
`--extra-density-model-name=_study_raw` to evaluate its raw comparison. The
reported training loss curves refer to EMA when enabled; per-batch optimizer
losses still describe the ordinary weights used to compute gradients.

Negative-weight training uses a batch-dependent safeguard: a negative event's NLL contribution is set to zero if its NLL exceeds the maximum NLL among positive events in that batch. Reported epoch evaluation losses use the unclipped signed objective. This stabilises a potentially unbounded signed-weight objective but can modify the training target.

## Buffered epoch shuffling

```yaml
shuffle_training: True
shuffle_buffer_size: 65536
shuffle_seed: 42
```

Buffered shuffling reads a sequential block of training rows, applies one shared permutation to features, conditions and weights, and yields optimiser batches from that block. It does not rewrite parquet files. With `batch_size: 2048`, a 65,536-row buffer holds 32 optimiser batches.

Each buffer is read directly from parquet using separate buffer-sized loaders. The optimiser batch size and evaluation loaders are unchanged, and only the current buffer is held in memory.

The buffer size must be an integer multiple of `batch_size` and at least one batch. The last buffer/batch can be smaller; every row is retained. Empty condition tables for unconditional models are supported.

The seed initialises one generator per training call. It advances across buffers and epochs, so epochs receive different permutations. Repeating the same run with the same data and seed reproduces the shuffle sequence. Mixing is limited to each sequential buffer; there is no carry-over pool between buffers.

Only optimisation batches are shuffled. Initial/epoch loss evaluation and validation continue reading the original data order. The worker default is `shuffle_training: False`; `configs/architecture/density_default.yaml` explicitly enables it. Older saved architectures without these keys retain the worker default.

## Fine-tuning and scans

```bash
innfer --cfg="configs/run/your_analysis.py" --step="TrainDensity" \
  --specific-file-name="ttbar" --specific-category="2223" \
  --density-architecture="configs/architecture/density_btm_2223_smaller_range_finetune.yaml" \
  --load-weights-for-training="models/BTM_Full_030926_smaller_range/density_ttbar_2223/ttbar.h5" \
  --extra-density-model-name="_finetune"
```

Use your matching run configuration and checkpoint. Fine-tuning can change learning rate, epochs and compatible training options, but changing coupling counts, widths, bins or condition-encoder shapes generally prevents loading the old weights. Preserve preprocessing transformations when reusing a model.

Grid scans enumerate candidate values. Bayesian scans treat integer/float lists as ranges and categorical lists as choices according to the tuning implementation. Do not pass a scan file directly to TrainDensity. If tuning `batch_size` while enabling buffered shuffling, every candidate must divide `shuffle_buffer_size`; the default Bayesian batch-size range does not guarantee this, so keep shuffling disabled or fix a compatible batch size for that scan.

[Back to all steps](steps.md).

## Standalone TensorFlow flow

Set `type: KerasFlow` to use
[KerasFlowNetwork](../python/worker/keras_flow_network.py). This backend
uses TensorFlow/Keras directly, without a BayesFlow model or trainer. Start with
[configs/architecture/density_keras_flow.yaml](../configs/architecture/density_keras_flow.yaml).
The existing `TrainDensity`, `EvaluateDensity`, `DensityPerformanceMetrics`,
`MakeAsimov`, generator, PValue and inference routes select it through
`InitiateDensityModel`. The BayesFlow-specific `Flow` animation is not supported.

Each coupling uses one conditioner to predict both affine parameters or all
spline parameters. This reduces conditioner evaluations compared with the
BayesFlow implementation, but speed and accuracy depend on the chosen topology
and hardware. Coupling counts do not represent equal capacity across backends.
No performance improvement on a particular dataset is implied.

| Key | Meaning |
| --- | --- |
| `coupling_design` | `affine`, `spline`, or alternating `interleaved`. |
| `num_coupling_layers` | Number of couplings when `coupling_layers` is null. |
| `hidden_units` | Conditioner hidden widths, e.g. `[128, 128]`. An empty list gives a linear conditioner. |
| `activation`, `dropout`, `residual` | Conditioner activation, training-only dropout probability, and residual additions between equally sized hidden representations. |
| `scale_clip` | Maximum absolute affine log-scale, bounded smoothly with tanh. |
| `permutation` | `random`, `reverse`, or `none`. Random permutations are fixed using `seed`. |
| `spline_bins`, `spline_bound` | Rational-quadratic spline bins and symmetric training-space interval. The transform is identity outside that interval. |
| `min_bin_size`, `min_slope` | Positive spline bin-size and derivative lower bounds. |
| `condition_dim` | Optional shared condition embedding width; null passes conditions directly. |
| `condition_hidden_units` | Hidden widths of the optional condition encoder. |
| `seed` | Network initialization and permutation seed. |
| `jit_compile` | Optional TensorFlow XLA compilation, disabled by default. |
| `clip_negative_weight_loss` | Enable the existing negative-weight safeguard in training; evaluation always reports the unclipped signed NLL. |

A `coupling_layers` list overrides the global coupling count/design and permits
per-layer overrides of the conditioner and transform settings:

```yaml
type: KerasFlow
permutation: random
coupling_layers:
  - design: affine
    hidden_units: [128, 128]
  - design: spline
    hidden_units: [192, 192, 192]
    spline_bins: 12
    residual: true
```

Training uses the repository optimizer/schedules, buffered shuffling, signed
weights, best evaluation checkpoint selection, optional epoch checkpoints, loss
plots and W&B logging. `l2_lambda`/`log_l2_lambda` apply only to optimization;
`Loss` excludes L2. AdamW retains its separate optimizer weight decay. Training
batches require positive net weight. BayesFlow-specific active learning,
resampling and per-epoch coupling/condition schedules are rejected when enabled.
Use this backend's architecture keys rather than BayesFlow's `affine_*`,
`spline_*` and summary-network keys.

The backend supports true one-dimensional flows without an auxiliary observable
or numerical normalization integral. Probability evaluation uses the established
physical-space transformations/Jacobians, including their existing restrictions
on derivatives of nonlinear preprocessing. Sampling uses the same empirical
range-filter convention as BayesFlow for fixed-condition physical samples;
`transform_X: false` or `apply_range_filter: false` preserves the full flow
support. Full-space normalization does not normalize these empirical range cuts.

Weights are saved as native Keras `.h5` files with the usual architecture YAML.
They cannot be loaded from BayesFlow checkpoints. Train a fresh model under a
separate suffix, then run the same validation commands with that suffix:

```bash
python3 scripts/innfer.py \
  --cfg=configs/run/btm_full_030926_sbi_smaller_range_balanced_split.py \
  --step=TrainDensity \
  --specific-file-name=ttbar --specific-category=2223 \
  --density-architecture=configs/architecture/density_keras_flow.yaml \
  --extra-density-model-name=_keras_flow \
  --save-model-per-epoch
```
