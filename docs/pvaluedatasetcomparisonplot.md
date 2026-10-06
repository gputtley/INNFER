---
layout: page
title: "Step: PValueDatasetComparisonPlot"
---

Compare simulation-versus-synthetic metrics with their synthetic-versus-synthetic null distribution.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PValueDatasetComparisonPlot"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`, `val_ind`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** PValueSimVsSynth metrics and collected PValueSynthVsSynth metrics.

**Produces:** p_value_dataset_comparison_{metric}.pdf plots and a dummy.txt completion marker.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/PValueDatasetComparisonPlot{extra_output_dir_name}{validation_suffix}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--pvalue-per-val-ind` | `False` | Run the p values per validation index |

## Behaviour and checks

The empirical upper-tail estimate is `(1 + number of toys with statistic >= observed)/(number of toys + 1)`. AUC should be judged against the matched toy distribution, whose centre need not be exactly 0.5. The smallest reportable p-value is 1/(N_toys+1).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/p_value_dataset_comparison_plot.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
