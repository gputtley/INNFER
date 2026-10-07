---
layout: page
title: "Step: DensityPerformanceMetrics"
---

Measure density loss, marginal distributions, multidimensional separation and optionally inference closure.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="DensityPerformanceMetrics"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`, `extra_name`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Density model weights/architecture, parameters.yaml, training/testing tables and the selected validation splits.

**Produces:** metrics.yaml under DensityPerformanceMetrics/{model_name}, plus generated samples and requested diagnostic plots.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/DensityPerformanceMetrics{extra_output_dir_name}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--cache-observable-transforms` | `False` | Cache parameter-independent density observable transforms and physical probability conversions during likelihood evaluation |
| `--density-correction` | `False` | Apply the classifier correction to generated comparisons and inference closure. |
| `--extra-density-correction-input-dir-name` | `''` | Suffix of the correction training output directories. |
| `--density-performance-metrics` | `'loss,histogram,multidim'` | Comma separated list of density performance metrics |
| `--density-performance-metrics-multidim` | `'bdt'` | Comma separated list of multidimensional density performance metrics |
| `--hold-dataset-in-memory` | `False` | Keep raw likelihood datasets in RAM and serve copies of the requested batches; default is streaming from parquet |
| `--loop-over-epochs` | `False` | Loop over epochs for performance metrics |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--number-of-integral-events` | `10 ** 5` | Integration events used to normalize the corrected closure-fit density. |

## Metric selection and generated samples

The default metric list is `loss,histogram,multidim`; inference closure requires adding `inference`. For example:

```bash
innfer --cfg="configs/run/your_analysis.py" --step="DensityPerformanceMetrics" \
  --specific-file-name="ttbar" --specific-category="2223" \
  --density-performance-metrics="loss,histogram,multidim,inference" \
  --hold-dataset-in-memory --cache-observable-transforms
```

The two cache flags affect the inference evaluation path. Generated samples are temporary in this step's default CLI configuration (`tidy_up_asimov` is True). The p-value steps retain their reference/toy samples for subsequent comparisons.

With `--density-correction`, both generation comparisons and inference closure use the correction trained for this density checkpoint. The closure likelihood normalizes the corrected shape at every condition; its integral uses fixed latent seeds. The `loss` metrics remain the original flow's training/test loss, so they can still be compared with its training objective.

Use `--loop-over-epochs` only when the corresponding epoch checkpoints exist; TrainDensity needs `--save-model-per-epoch` to produce them. Keep the saved model architecture and preprocessing metadata with each checkpoint.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/density_performance_metrics.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
