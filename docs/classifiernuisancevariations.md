---
layout: page
title: "Step: ClassifierNuisanceVariations"
---

Evaluate classifier shape effects against simulation nuisance variations.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ClassifierNuisanceVariations"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `nuisance`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Trained classifier, parameters.yaml and nominal/up/down simulation tables.

**Produces:** nuisance_variations_performance_metrics_{parameter}.yaml and classifier_nuisance_variations PDFs.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Plots | `$PLOTS_DIR/$CFG_NAME/ClassifierNuisanceVariations{extra_output_dir_name}/{process}/{category}` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/ClassifierNuisanceVariations{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--classifier-divide-by-nominal` | `False` | Divide the classifier by the nominal value |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/classifier_nuisance_variations.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
