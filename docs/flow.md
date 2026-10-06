---
layout: page
title: "Step: Flow"
---

Visualise how the learned flow transforms observables through its coupling layers.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="Flow"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Trained density model and preprocessing metadata/data.

**Produces:** flow_{column}_cl{layer}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Plots | `$PLOTS_DIR/$CFG_NAME/Flow{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/flow.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
