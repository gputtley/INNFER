---
layout: page
title: "Step: PValueSynthVsSynthCollect"
---

Collect the synthetic-comparison toy metrics into one null distribution.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PValueSynthVsSynthCollect"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`, `val_ind`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** All requested PValueSynthVsSynth toy metric files.

**Produces:** metrics.yaml under PValueSynthVsSynthCollect{suffix}/{model_name}.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/PValueSynthVsSynthCollect{extra_output_dir_name}{validation_suffix}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--number-of-toys` | `100` | The number of toys for p-value dataset comparisons |
| `--pvalue-per-val-ind` | `False` | Run the p values per validation index |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/p_value_synth_vs_synth_collect.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
