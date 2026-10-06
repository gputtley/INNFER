---
layout: page
title: "Step: NormalisationSplineClassifier"
---

Integrate classifier ratio predictions over generated density samples and fit a normalisation spline.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="NormalisationSplineClassifier"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** MakeAsimovForNormalisationSpline samples, the trained classifier and metadata.

**Produces:** {process}_norm_spline.pkl beside the classifier and norm_spline_{parameter}.pdf.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/NormalisationSplineClassifier/{model_name}{extra_classifier_model_name}` |
| Models read | `$MODELS_DIR/$CFG_NAME` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/normalisation_spline_classifier.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
