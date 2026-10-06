---
layout: page
title: "Step: CompareConstraintsAndImpacts"
---

Compare constraints and impacts from multiple fit configurations.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="CompareConstraintsAndImpacts"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `freeze_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Completed fit/impact summaries selected by the comparison directory names.

**Produces:** Constraint/impact comparison plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/CompareConstraintsAndImpacts{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--compare-constraints-and-impacts-input` | `None` | Compare the constraints and impacts from this extra directory |
| `--compare-constraints-and-impacts-names` | `None` | Names of the constraints and impacts to compare |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/compare_constraints_and_impacts.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
