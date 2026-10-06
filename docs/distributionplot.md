---
layout: page
title: "Step: DistributionPlot"
---

Compare the fitted dataset with prefit or post-fit predictions, optionally showing uncertainty bands.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="DistributionPlot"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Selected observed/simulation tables, prefit or post-fit samples, fit metadata and optional uncertainty samples.

**Produces:** Generator-style distribution PDFs for unbinned fits or binned_distribution_category PDFs for binned fits.

| Location | Path pattern |
| --- | --- |
| Plots | `$PLOTS_DIR/$CFG_NAME/DistributionPlot{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

`--summary-from` selects the upstream result family (for example Scan, Covariance or CovarianceWithDMatrix). Keep its directory suffix and validation indexing consistent with the producing steps.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--binned-observed-from-predicted` | `False` | Take the binned observed data from the predicted values and not the validation samples |
| `--data-vs-simulation` | `False` | For the Generator and DistributionPlot step, show data vs simulation |
| `--include-uncertainty` | `False` | Include the postfit uncertainties in the postfit plots. |
| `--likelihood-type` | `'unbinned_extended'` | Type of likelihood to use for fitting. |
| `--no-constraint` | `False` | Do not use the constraints |
| `--plot-2d-unrolled` | `False` | Make 2D unrolled plots when running generator. |
| `--plot-extra-hypothesis` | `None` | For Generator steps (and binned postfit), plot extra hypotheses. This is semi colon separated, comma separated key=value inputs |
| `--plot-var-and-bins` | `None` | For Generator steps, variable name and string of bins with () or [] depending on if you want equally spaced. |
| `--prefit-nuisance-constraints` | `False` | Make postfit plots with prefit nuisance constraints |
| `--prefit-nuisance-values` | `False` | Make postfit plots with prefit nuisance values |
| `--ratio-range` | `'0.5,1.5'` | Range for ratio plot |
| `--use-expected-data-uncertainty` | `False` | In postfit plots change the data uncertainty to the expected stat uncertainty |
| `--use-prefit-asimov` | `False` | Use prefit asimov when running DistributionPlot. |

## Behaviour and checks

This is the implemented CLI step for prefit/postfit distribution comparisons. The old documentation name PostFitPlot is not an implemented step selector. Use `--use-prefit-asimov` for prefit predictions and `--include-uncertainty` for generated uncertainty variations.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/generator.py), [Runner](../python/runner/binned_distributions.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
