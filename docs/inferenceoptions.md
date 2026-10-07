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
| `--density-correction` | Off | Apply the trained conditional simulation-to-flow correction in unbinned densities and normalize the full shape. |
| `--extra-density-correction-input-dir-name` | `''` | Select correction artifacts trained with an output-directory suffix. |
| `--use-spline` | Off | Enable saved classifier/regression normalisation splines in generation, calibration and inference. They are skipped by default; the legacy `--no-spline` flag remains accepted. |
| `--include-per-model-rate`, `--include-per-model-lnN` | Off | Include these terms in individual-process likelihoods as well as the combined setup. |
| `--keep-lnN-if-rate-param` | Off | Retain the configured lnN effects when a process has a floating rate. |

Use `--specific-file-name=combined` for the combined likelihood when the configuration defines more than one process. Categories can be selected with `--specific-category`. Dataset and model suffixes are described in [common step options](stepoptions.md).

Normalisation splines are skipped unless `--use-spline` is supplied. In a Snakemake step's `run_options`, set `use_spline: True` to enable them. Enabling this option also declares the saved `_norm_spline.pkl` files as dependencies. The legacy `--no-spline` flag remains accepted; it cannot be combined with `--use-spline`.

For numerical comparisons, distinguish simulation sum-of-weights, effective event count and generated Monte Carlo event count. Increasing synthetic statistics does not increase the simulated sample's independent information.

## Minimisation and integration

`--minimisation-method` defaults to `scipy`. `--initial-best-fit-guess` changes the starting parameter point, and `--simplex` supplies simplex settings in the supporting likelihood path. These are separate from frozen parameter values. `--no-likelihood-print-out` suppresses likelihood evaluation output.

Use `--minimisation-method=gradient-descent` for pure steepest descent with numerical likelihood gradients. Each gradient uses exactly two likelihood evaluations per free parameter, with a single central difference and no extrapolation sweep. The difference increment is `float32_epsilon**(1/3) * parameter_scale`, reflecting the flow's float32 evaluation precision. Parameter scales use the smallest available density-model training standard deviation for that condition, falling back to `max(1, abs(initial_value))`. This is a precision-based heuristic rather than an adaptive error estimate. Each update follows the negative numerical gradient, with an optimisation step size chosen separately by SciPy's Wolfe line search and Armijo backtracking as a fallback; line search can request additional gradients. It supports frozen parameters and profiled scans, uses the ordinary likelihood evaluation batch size, and stops after two consecutive accepted improvements of at most `0.01` in the absolute objective (`-2 ln L`). The likelihood attribute `gradient_descent_objective_tolerance` controls this threshold. A maximum absolute gradient of `1e-6` also permits an earlier stationary-point exit, and the iteration limit remains 1000. Small successive changes are a convergence criterion, not a guaranteed bound on the distance from the true minimum. A warning reports failure to converge; no momentum or Hessian approximation is used.

Ratio reintegration uses `--number-of-integral-events` (default 100,000) and the supporting integral-scaling configuration. Increasing the integration sample improves numerical precision at a computational cost; use identical integration settings when comparing fit stages.

With `--density-correction`, each process/category uses its `DensityCorrectionWithClassifier` checkpoint. The classifier evaluates physical observables and conditions using its saved density transforms. Its simulation-to-flow odds multiply the density together with any nuisance ratios, and the complete shape is normalized by integrating over the original flow. This normalization also runs with `--only-density` and does not require `--integrate-density-with-ratios`. Process yields and their relative mixture weights remain the configured yield predictions.

Pass the same correction and model suffixes to generation, fits, scans, Hessians, DMatrix and calibration. Empirical training minima and maxima do not impose hard support cuts on the corrected likelihood. Its BayesFlow normalizer samples the original flow without those cuts; ordinary generated samples retain their existing range filtering. Gradients and Hessians of the corrected density use central differences of the full normalized log density, including the condition-dependent normalizer, with fixed latent seeds for integration and a step of 0.01 in physical parameter units. These evaluations cost more than the original analytical derivative path. Binned and Poisson-only fits use their existing bin/yield predictions rather than the event density, so this flag does not change those predictions.

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
