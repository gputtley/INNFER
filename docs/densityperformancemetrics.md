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

**Produces:** `metrics.yaml`, a standalone LaTeX report `metrics.tex` and its compiled `metrics.pdf` under DensityPerformanceMetrics/{model_name}, plus generated samples and requested diagnostic plots. Epoch and result suffixes apply to all three files, for example `metrics_epoch_2.yaml`, `metrics_epoch_2.tex` and `metrics_epoch_2.pdf`.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/DensityPerformanceMetrics{extra_output_dir_name}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## LaTeX and PDF tables

The report is generated automatically with the same command as the metrics. Compact matrices place validation hypotheses in columns and observables or metrics in rows, grouped by dataset. Column headings show the translated hypothesis and value, including units, instead of validation indices. Dataset labels display `train_inf` as `Train`, `test_inf` as `Test`, `val` as `Validation` and `full` as `Full`; the YAML keys remain unchanged. Losses share one row, and dataset-wide aggregates share an `All / merged` column. Process, category, observable and parameter labels use `Translate`. Wide scans split into groups of at most six columns; long tables continue across pages with repeated headings. Additional metrics use paired metric/value columns.

Histogram cells show chi-squared/ndf followed by `(raw chi-squared/ndf)` when those values are available. Inference closure pull-squared has a separate label. Mean/sum histogram rows aggregate individual observable scores rather than forming a pooled chi-squared. Losses remain those of the original flow, including when density correction is enabled. Only available metric groups are displayed. The YAML retains full precision; tables show four to seven significant figures and explicitly mark missing or non-finite values.

PDF generation requires `pdflatex` on the execution host, with the `booktabs`, `longtable`, `array` and `geometry` packages. The report is compiled in a temporary directory with shell execution disabled. If compilation fails, the step reports an error and keeps the YAML, LaTeX and a `metrics_latex.log` diagnostic (with matching suffixes). Internal Bayesian tuning trials retain YAML-only output to avoid compiling reports for every trial; evaluating the selected model with DensityPerformanceMetrics produces the tables.

The `.tex` file is a complete document and can also be compiled independently:

```bash
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error metrics.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error metrics.tex
```

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--cache-observable-transforms` | `False` | Cache parameter-independent density observable transforms and physical probability conversions during likelihood evaluation |
| `--density-correction` | `False` | Apply the classifier correction to generated comparisons and inference closure. |
| `--extra-density-correction-input-dir-name` | `''` | Suffix of the correction training output directories. |
| `--density-performance-metrics` | `'loss,histogram,multidim'` | Comma separated list of density performance metrics |
| `--density-performance-metrics-inference-datasets` | `'test_inf,val'` | Comma separated closure-fit splits: `train_inf`, `test_inf`, `val`, `full`. |
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

When looping over epochs, generated-sample folders include the checkpoint
suffix, for example `val_ind_2_epoch_1_seed_42_for_val`. Each checkpoint reads
and cleans up its own samples, so parallel checkpoint jobs do not share them.
Histogram/BDT results produced before this checkpoint-isolation fix should be
regenerated if different epochs ran concurrently in the same output directory.

To compare inference closure on training events and held-out samples, use:

```bash
innfer --cfg="configs/run/your_analysis.py" --step="DensityPerformanceMetrics" \
  --density-performance-metrics="loss,inference" \
  --density-performance-metrics-inference-datasets="train_inf,test_inf,val" \
  --cache-observable-transforms
```

`train_inf` selects the physical, hypothesis-specific training-event samples
under `PreProcess/{process}/{category}/val_ind_{i}/`, with their existing event
weights and effective-event normalization. It does not fit the transformed,
random-condition copies in the density training table. Each requested split
must have `X`, `Y` and `wt` files and corresponding `eff_events` metadata.
`Inputs()` includes the selected closure files for dependency tracking.
The option changes inference closure only; loss, histogram and multidimensional
dataset selections retain their existing defaults. Results are stored together
with keys such as `inference_best_fit_train_inf_val_ind_2_bw_mass`.
The `inference_*_mean` entries average over the selected available splits;
adding Train therefore changes those means. Train closure is a diagnostic of
seen-event behaviour, not an independent generalization test.

Add `--density-performance-metrics-per-val-ind` to evaluate multidimensional metrics separately for every eligible validation index instead of combining the hypotheses. All results are written to the same `metrics.yaml`, with keys such as `bdt_auc_val_val_ind_2` and `bdt_signed_auc_test_inf_val_ind_2`. Each comparison uses that index's effective event count and excludes the fixed condition columns from its classifier inputs. Empty datasets and non-density hypotheses are skipped. Without the flag, the existing merged multidimensional metrics are retained. Histogram and inference closure metrics already include validation indices; training/test losses remain global.

With `--density-correction`, both generation comparisons and inference closure use the correction trained for this density checkpoint. The closure likelihood normalizes the corrected shape at every condition; its integral uses fixed latent seeds. The `loss` metrics remain the original flow's training/test loss, so they can still be compared with its training objective.

Use `--loop-over-epochs` only when the corresponding epoch checkpoints exist; TrainDensity needs `--save-model-per-epoch` to produce them. Epoch weights such as `ttbar_epoch_2.h5` use the shared `ttbar_architecture.yaml` in the same model directory, together with the original preprocessing metadata. Select one epoch with `--specific="extra_name=_epoch_2"`. Its results are saved as `metrics_epoch_2.yaml`.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/density_performance_metrics.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
