---
layout: page
title: "Step: MakeAsimovNuisanceDoubleVariations"
---

Generate synthetic samples with pairs of nuisances shifted simultaneously.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="MakeAsimovNuisanceDoubleVariations"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `nuisance_1`, `nuisance_2`, `nuisance_1_shift`, `nuisance_2_shift`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Density/ratio models and metadata for the selected nuisance pairs.

**Produces:** asimov.parquet under the paired-variation directories in MakeAsimovNuisanceDoubleVariations.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--asimov-seed` | `42` | The seed to use the create the asimov |
| `--classifier-divide-by-nominal` | `False` | Divide the classifier by the nominal value |
| `--no-spline` | `False` | Do not use the normalisaing splines when creating asimov |
| `--number-of-asimov-events` | `10 ** 6` | The number of asimov events |
| `--only-density` | `False` | Build asimov from only the density model |
| `--skip-weight-variation` | `False` | For double variations skip the weight variation |
| `--use-asimov-scaling` | `10` | Generate asimov with this scaling up of the predicted yield |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/make_asimov.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
