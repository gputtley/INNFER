---
layout: page
title: "Step: PValueSimVsSynth"
---

Measure multidimensional differences between simulation and samples from the density model.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PValueSimVsSynth"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`, `val_ind`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Density model and simulation validation datasets; no PValueSynthVsSynth output is needed for this stage.

**Produces:** metrics.yaml and generated samples under PValueSimVsSynth{suffix}/{model_name}.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/PValueSimVsSynth{extra_output_dir_name}{validation_suffix}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--density-correction` | `False` | Apply the trained density correction to synthetic event weights. |
| `--extra-density-correction-input-dir-name` | `''` | Suffix of the correction training output directories. |
| `--density-performance-metrics-multidim` | `'bdt'` | Comma separated list of multidimensional density performance metrics |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--pvalue-per-val-ind` | `False` | Run the p values per validation index |
| `--pvalue-per-val-ind-hypothesis` | `None` | The hypothesis to use for p value calculation per validation index |

## Behaviour and checks

This step measures a test statistic, not a p-value by itself. Use the same hypotheses, metrics, dataset split and model selection throughout the four-stage [p-value workflow](pvalueworkflow.md). Density-only generation is used here; this does not test the complete nuisance-ratio likelihood.

With `--density-correction`, the classifier ratio multiplies the original simulation weights assigned to generated rows. Run `PValueSimVsSynth` and `PValueSynthVsSynth` with the same flag and correction checkpoint; both null samples then use the corrected distribution. The cached reference must contain the matching `density_correction.yaml` sidecar. Rerun the simulation comparison before generating corrected null toys.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/density_performance_metrics.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
