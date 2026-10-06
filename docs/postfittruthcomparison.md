---
layout: page
title: "Step: PostFitTruthComparison"
---

Compare synthetic predictions at the truth and fitted hypotheses.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PostFitTruthComparison"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Truth MakeAsimov samples, MakePostFitAsimov samples and the fitted dataset.

**Produces:** postfit_truth_comparison.pdf and per-observable efficiency plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/PostFitTruthComparison{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/postfit_truth_comparison.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
