---
layout: page
title: "Step: SummaryNuisanceVariations"
---

Summarise fitted shifts and closure for nuisance-varied validation samples.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="SummaryNuisanceVariations"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Fit/uncertainty summaries produced with validation-loop-over-nuisance-variations.

**Produces:** summary_nuisance_variations_page{page}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/SummaryNuisanceVariation{extra_output_dir_name}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

`--summary-from` selects the upstream result family (for example Scan, Covariance or CovarianceWithDMatrix). Keep its directory suffix and validation indexing consistent with the producing steps.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/summary_nuisance_variations.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
