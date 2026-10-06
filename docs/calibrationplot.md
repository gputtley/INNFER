---
layout: page
title: "Step: CalibrationPlot"
---

Compare learned likelihood-ratio predictions between a validation hypothesis and the default simulation sample.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="CalibrationPlot"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Density model and the selected simulation split at both hypotheses.

**Produces:** calibration_{validation_index}.yaml and corresponding PDFs under Calibration.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/Calibration{extra_output_dir_name}/{process}/{category}` |
| Plots | `$PLOTS_DIR/$CFG_NAME/Calibration{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--calibration-n-bins` | `30` | Number of likelihood ratio bins to use in the Calibration step. |

## Behaviour and checks

The default hypothesis is the reference sample and is skipped as a numerator hypothesis. `--sim-type` selects the simulation split; a ratio-calibration plot complements marginal generator plots and multidimensional separation tests.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/calibration.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
