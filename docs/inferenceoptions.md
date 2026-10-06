---
layout: page
title: "Inference options"
---

Likelihood stages share data selection, model selection, parameter freezing and evaluation settings. Keep these settings identical when producing a best fit, its uncertainty and a profile scan.

## Data and likelihood selection

| Option | Default | Meaning |
| --- | --- | --- |
| `--data-type` | `sim` | `sim`: preprocessed simulation; `asimov`: generated synthetic events; `data`: DataCategories observed events. |
| `--sim-type` | `val` | Simulation split, such as `val`, `test_inf`, `train_inf` or `full`. Training `test` tables are transformed model-training tables and are not interchangeable with physical `test_inf` tables. |
| `--likelihood-type` | `unbinned_extended` | `unbinned`, `unbinned_extended`, `binned`, `binned_extended` or `poisson`. Binned modes require prepared bin-yield metadata. |
| `--no-constraint` | Off | Omit auxiliary constraint terms. |
| `--scale-to-eff-events` | Off | Use effective simulation statistics instead of the nominal yield scaling. |
| `--integrate-density-with-ratios` | Off | Reintegrate the density after likelihood-ratio corrections using generated integration events. |
| `--classifier-divide-by-nominal` | Off | Express the classifier correction relative to its nominal prediction. |
| `--use-spline` | Off | Enable saved classifier/regression normalisation splines in generation, calibration and inference. They are skipped by default; the legacy `--no-spline` flag remains accepted. |
| `--include-per-model-rate`, `--include-per-model-lnN` | Off | Include these terms in individual-process likelihoods as well as the combined setup. |
| `--keep-lnN-if-rate-param` | Off | Retain the configured lnN effects when a process has a floating rate. |

Use `--specific-file-name=combined` for the combined likelihood when the configuration defines more than one process. Categories can be selected with `--specific-category`. Dataset and model suffixes are described in [common step options](stepoptions.md).

Normalisation splines are skipped unless `--use-spline` is supplied. In a Snakemake step's `run_options`, set `use_spline: True` to enable them. Enabling this option also declares the saved `_norm_spline.pkl` files as dependencies. The legacy `--no-spline` flag remains accepted; it cannot be combined with `--use-spline`.

For numerical comparisons, distinguish simulation sum-of-weights, effective event count and generated Monte Carlo event count. Increasing synthetic statistics does not increase the simulated sample's independent information.

## Minimisation and integration

`--minimisation-method` defaults to `scipy`. `--initial-best-fit-guess` changes the starting parameter point, and `--simplex` supplies simplex settings in the supporting likelihood path. These are separate from frozen parameter values. `--no-likelihood-print-out` suppresses likelihood evaluation output.

Ratio reintegration uses `--number-of-integral-events` (default 100,000) and the supporting integral-scaling configuration. Increasing the integration sample improves numerical precision at a computational cost; use identical integration settings when comparing fit stages.

## Freezing parameters

`--freeze="nuisance=0,another=1"` fixes named parameters to explicit values. Symbolic selections include:

- `all-nuisances`: freeze parameters listed as nuisances in the configuration.
- `all-but-one`: create separate configurations with one selected parameter floating.
- `all-but-{parameter}`: leave the named parameter floating.
- `all-but-varied`: leave the validation-varied parameters floating.
- `all-non-density`: freeze parameters not used as density conditions.

Selection and directory suffixes depend on the configured parameter loops. Without a loaded full fit, symbolic freezes use the validation/default parameter values. `all-nuisances` does not automatically freeze separately configured rate parameters unless they are included in that nuisance selection.

### Freeze at a previous full-fit result

Use `--load-fit-for-defaults` alongside a symbolic freeze. It accepts either a best-fit YAML path or an InitialFit directory suffix:

```bash
innfer --cfg="configs/run/your_analysis.py" --step="InitialFit" \
  --data-type="data" --specific-file-name="combined" --extra-dir-name="Data"

innfer --cfg="configs/run/your_analysis.py" --step="InitialFit" \
  --data-type="data" --specific-file-name="combined" \
  --freeze="all-nuisances" --load-fit-for-defaults="Data" \
  --extra-dir-name="DataStat"
```

For a suffix `Data`, each iteration resolves `$EVAL_DATA_DIR/$CFG_NAME/InitialFitData/{process}/best_fit_{val_ind}.yaml`. A direct `.yaml` or `.yml` path selects that exact file. The input must contain aligned `columns` and `best_fit` arrays with finite values for the frozen parameters.

The fit file is a declared dependency. Runtime loading allows Snakemake to build the downstream graph before the full fit exists. Explicit numeric freezes remain explicit; loading a full fit is not a request to refreeze every fitted parameter or to change the starting minimiser guess.

Pass the same freeze and loaded-fit options to Hessian, covariance, scans and supporting stages. Use `--initial-best-fit-guess` separately when changing the fit's starting point.

## Optional evaluation caches

Both flags default to off and are supported by inference and the inference part of [DensityPerformanceMetrics](densityperformancemetrics.md):

- `--hold-dataset-in-memory` keeps raw likelihood datasets in RAM. Batch consumers receive copies of the requested rows, followed by the usual selections/transforms/weight handling. Memory use grows with the loaded datasets.
- `--cache-observable-transforms` caches density transforms and conversion factors that do not depend on the current fit parameters. Condition-dependent network evaluations still run at each parameter point. The cache is bounded and invalidates when transform metadata changes.

These options reduce repeated preparation work; they do not change the likelihood definition or uncertainty prescription. Choose them according to available memory and reuse across evaluations.

## Typical uncertainty chains

```text
InitialFit → Hessian → Covariance
InitialFit → Hessian + DMatrix → CovarianceWithDMatrix
InitialFit → HessianParallel → HessianCollect → Covariance
InitialFit → HessianNumericalParallel → HessianCollect → Covariance
InitialFit → Hessian → ScanPointsFromHessian → Scan → ScanCollect → ScanPlot
```

Weighted samples can require the sandwich covariance `H⁻¹ D H⁻¹`; see [statistical methods](statistics.md). A successful covariance calculation does not establish model closure.

[Back to all steps](steps.md).
