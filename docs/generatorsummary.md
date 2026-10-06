---
layout: page
title: "Step: GeneratorSummary"
---

Summarise generator comparisons across validation hypotheses.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="GeneratorSummary"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** PreProcess simulation and MakeAsimov samples at all selected validation indices.

**Produces:** generation_summary_{column}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/GeneratorSummary{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--use-scenario-labels` | `False` | Use Scenario 1, for example, labelling on plots rather than the string name |
| `--val-inds` | `None` | val_inds for summary plots. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/generator_summary.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
