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

For `CosineDecayWithWarmUp`, `warmup_fraction` sets the initial ramp, `step_power` changes the progress through the cosine, and this repository's `alpha` is an absolute terminal learning rate. `step_power: 1` gives ordinary cosine progress after warmup; a value below one decays faster early in that interval. Other schedulers have their own parameter conventions; see [optimizer.py](../python/worker/optimizer.py).

The trainer saves the checkpoint with the lowest finite evaluation loss, including epoch zero. Fine-tuning therefore preserves the loaded starting weights when no epoch improves the loss. `--save-model-per-epoch` also writes `_epoch_{index}.h5` checkpoints.

Negative-weight training uses a batch-dependent safeguard: a negative event's NLL contribution is set to zero if its NLL exceeds the maximum NLL among positive events in that batch. Reported epoch evaluation losses use the unclipped signed objective. This stabilises a potentially unbounded signed-weight objective but can modify the training target.

## Buffered epoch shuffling

```yaml
shuffle_training: True
shuffle_buffer_size: 65536
shuffle_seed: 42
```

Buffered shuffling reads a sequential block of training rows, applies one shared permutation to features, conditions and weights, and yields optimiser batches from that block. It does not rewrite parquet files. With `batch_size: 2048`, a 65,536-row buffer holds 32 optimiser batches.

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
