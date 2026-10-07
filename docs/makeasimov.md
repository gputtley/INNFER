---
layout: page
title: "Step: MakeAsimov"
---

Generate finite weighted synthetic samples at configured validation hypotheses.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="MakeAsimov"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Trained density and applicable ratio models, metadata, and any normalisation splines or pruning metrics requested.

**Produces:** MakeAsimov/{process}/{category}/val_ind_{index}/asimov.parquet.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/MakeAsimov{extra_output_dir_name}/{process}/{category}/val_ind_{val_ind}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--classifier-divide-by-nominal` | `False` | Divide the classifier by the nominal value |
| `--density-correction` | `False` | Multiply generated weights by the trained simulation-to-flow classifier ratio. |
| `--extra-density-correction-input-dir-name` | `''` | Suffix used when training the correction with `--extra-output-dir-name`. |
| `--use-spline` | `False` | Opt in to saved classifier/regression normalisation splines. |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--only-default-asimov` | `False` | Build asimov for only the default validation indices |
| `--only-density` | `False` | Build asimov from only the density model |
| `--prune-classifier-models` | `None` | Comma separated list of key>values keep shape effects for |
| `--prune-from` | `'EvaluateClassifier'` | Step to prune from |
| `--use-asimov-scaling` | `10` | Generate asimov with this scaling up of the predicted yield |

## Behaviour and checks

These are finite Monte Carlo samples with weights, rather than noiseless event-level datasets. The workflow passes yield-based `--use-asimov-scaling` (default 10); the runner uses its fixed event count only when `use_asimov_scaling` is None. The CLI scaling option is integer-valued and takes precedence over the fixed-count option in this dispatch. `--only-density` omits learned nuisance-ratio corrections; `--use-spline` enables their saved normalisation splines; these are skipped by default. Keep these choices consistent with the validation/inference target.

With `--classifier-divide-by-nominal`, classifiers whose resolved evaluation parameter is zero are automatically skipped, including their model, spline and pruning dependencies.

With `--density-correction`, the correction classifier is loaded from `$MODELS_DIR/$CFG_NAME/DensityCorrectionWithClassifier{extra_density_correction_input_dir_name}/{density_model_name}{extra_density_model_name}{extra_classifier_model_name}/{process}.h5`. Its `parameters.yaml` is read from the matching directory under `$EVAL_DATA_DIR/$CFG_NAME`. Train it first using `DensityCorrectionWithClassifier` for the same density checkpoint and preprocessing transforms.

The classifier evaluates observables and conditions in its saved density training space. Its odds `P(simulation)/P(synthetic)` multiply event weights before nuisance-model weights. This also applies with `--only-density`. The final sample is still normalized to the predicted process yield; the correction changes its shape. A `density_correction.yaml` sidecar records the correction inputs. Use the same flag and checkpoint in unbinned inference to evaluate the matching normalized corrected density.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/make_asimov.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
