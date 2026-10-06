---
layout: page
title: "Step: PlotFactorisation"
---

Summarise metrics for single and paired nuisance variations to inspect factorisation approximations.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PlotFactorisation"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`, `nuisance_1_shift`, `nuisance_2_shift`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** ValidationPerformanceMetrics results for the configured variation scenarios.

**Produces:** FactorisationMatrix_{metric}.pdf and FactorisationMatrixStyle2_{metric}.pdf.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/PlotFactorisation{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--plot-metric` | `'chi_squared_per_dof:mean'` | Metric for factorisation plots, colon separated in many keys |
| `--skip-weight-variation` | `False` | For double variations skip the weight variation |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/plot_factorisation.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
