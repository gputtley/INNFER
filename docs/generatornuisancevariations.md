---
layout: page
title: "Step: GeneratorNuisanceVariations"
---

Plot nominal and up/down synthetic nuisance effects.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="GeneratorNuisanceVariations"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `nuisance`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Nominal MakeAsimov and MakeAsimovNuisanceVariations samples, metadata.

**Produces:** nuisance_distribution_{nuisance}_{column}.pdf plots.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/GeneratorNuisanceVariations{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/generator_nuisance_variations.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
