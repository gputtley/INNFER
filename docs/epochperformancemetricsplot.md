---
layout: page
title: "Step: EpochPerformanceMetricsPlot"
---

Plot saved performance metrics across model-training epochs.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="EpochPerformanceMetricsPlot"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Per-epoch checkpoints and metrics produced with the corresponding epoch loop.

**Produces:** Plots under EpochPerformanceMetricsPlot and an epoch_pm.txt completion marker.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/EpochPerformanceMetricsPlot{extra_output_dir_name}/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--other-input` | `None` | Other inputs to likelihood and summary plotting |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/epoch_performance_metrics_plot.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
