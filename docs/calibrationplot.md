---
layout: page
title: "Step: CalibrationPlot"
---

Compare the full event likelihood ratio between a validation hypothesis and the category's default hypothesis against the ratio measured from simulation. Each job is identified by `file_name`, `val_ind` and `category`; individual processes and `combined` are supported, including classifier and regression shape corrections.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="CalibrationPlot"
```

For the combined BTM likelihood in 2223, using the same corrections as the generation and inference workflows:

```bash
python3 scripts/innfer.py \
  --cfg="configs/run/btm_full_030926_sbi_smaller_range.py" \
  --step="CalibrationPlot" --specific-file-name="combined" \
  --specific-category="2223" --specific="val_ind=2" --sim-type="val" \
  --classifier-divide-by-nominal --integrate-density-with-ratios \
  --prune-classifier-models="q75:1.01" \
  --prune-from="ClassifierNuisanceVariationsTestInf"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `val_ind`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Density models, configured classifier/regression models, process metadata and the selected simulation split at both hypotheses. Saved normalisation splines are dependencies only when `--use-spline` is set; classifier pruning adds the configured pruning metric files. Model suffixes apply to both loading and declared dependencies.

**Produces:** `calibration_plot_{val_ind}{extra_plot_name}.yaml` and the corresponding PDF under CalibrationPlot. Nuisance-variation mode uses `{nuisance}_{up/down}` instead of the validation index. The YAML records bin edges, predicted and MC-estimated ratios and uncertainties, both parameter hypotheses, process/category identifiers and likelihood type.

| Location | Path pattern |
| --- | --- |
| Models read | `$MODELS_DIR/$CFG_NAME` |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/CalibrationPlot{extra_output_dir_name}/{process}/{category}` |
| Plots | `$PLOTS_DIR/$CFG_NAME/CalibrationPlot{extra_output_dir_name}/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--calibration-plot-n-bins` | `30` | Number of likelihood ratio bins to use in the CalibrationPlot step. |

The step shares inference's model settings: `--only-density`, `--classifier-divide-by-nominal`, `--integrate-density-with-ratios`, `--number-of-integral-events`, `--use-integral-scaling`, `--use-spline`, classifier/lnN pruning, per-model rate/lnN inclusion, and density/classifier/regression model suffixes. See [inference options](inferenceoptions.md) and [common step options](stepoptions.md). Both optional evaluation caches are supported.

Use `--validation-loop-over-nuisance-variations` for the dedicated up/down simulation datasets. Nuisance-only validation hypotheses are included by default; `--skip-non-density` restores the density-only hypothesis filter.

## Behaviour and checks

The default hypothesis is the reference sample and is skipped as a test hypothesis. `--sim-type` selects the simulation split. CalibrationPlot always uses simulation at both hypotheses, independently of `--data-type`. In `combined`, process-specific validation indices select the corresponding sample for each process, and all process contributions are evaluated for each event.

CalibrationPlot always uses the normalised unbinned likelihood, independently of the global `--likelihood-type` setting. The combined density is `p(x|H) = sum_process yield_process(H) * pdf_process(x|H) / sum_process yield_process(H)`. Relative process yields remain in the mixture; the overall expected yield cancels. Each simulation histogram is normalised by its full sample's total signed weight, including events outside the plotted range.

The calibration uses physical relative yields even with `--scale-to-eff-events`: a common effective-event factor cancels from the density ratio. Global Poisson and auxiliary constraint terms are not part of this per-event ratio.

With `--classifier-divide-by-nominal`, classifiers whose parameter is zero (the nominal classifier denominator point) at both H0 and H1 are automatically omitted, including their model, spline and pruning dependencies. Classifiers shifted in either hypothesis remain loaded; regression models and process yield effects remain unchanged.

Bins are equally spaced over the central reference-sample range. Histogram uncertainties use the sum of squared event weights and are propagated into the ratio; bins with non-positive test or reference contents have no MC ratio. A ratio-calibration plot complements marginal generator plots and multidimensional separation tests.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/calibration_plot.py), [likelihood implementation](../python/worker/likelihood.py). The runner inherits inference's model and yield builders. Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph. The [BTM calibration subworkflow](../configs/snakemake/subworkflow/calibration_plot_condor_ic.yaml) supplies the generation/inference correction settings for both `val` and `full` samples.

[Back to all steps](steps.md).
