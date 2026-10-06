---
layout: page
title: "Step: PValueSynthVsSynth"
---

Generate independent synthetic comparison toys to calibrate the multidimensional metrics.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PValueSynthVsSynth"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`, `toy`, `val_ind`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** The selected density model and the PValueSimVsSynth reference samples used by this workflow.

**Produces:** metrics_toy_{index}.yaml and toy samples under PValueSynthVsSynth{suffix}/{model_name}.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/PValueSynthVsSynth{extra_output_dir_name}{validation_suffix}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--density-performance-metrics-multidim` | `'bdt'` | Comma separated list of multidimensional density performance metrics |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--number-of-toys` | `100` | The number of toys for p-value dataset comparisons |
| `--pvalue-per-val-ind` | `False` | Run the p values per validation index |
| `--pvalue-per-val-ind-hypothesis` | `None` | The hypothesis to use for p value calculation per validation index |

## Behaviour and checks

Toys use alternative generation seeds; this is not a bootstrap of the observed simulation sample. Its reference synthetic sample comes from PValueSimVsSynth. Keep the toy model, sample size and hypothesis consistent with that reference.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/density_performance_metrics.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
