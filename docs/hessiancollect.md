---
layout: page
title: "Step: HessianCollect"
---

Assemble the Hessian from parallel matrix-entry files.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="HessianCollect"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `freeze_ind`, `val_ind`, `nuisance`, `variation`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** HessianParallel or HessianNumericalParallel entry files for the requested columns.

**Produces:** hessian_{validation_index}.yaml under Hessian.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/Hessian{extra_output_dir_name}{freeze_suffix}/{process}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

See [Inference options](inferenceoptions.md) for data selection, likelihood types, frozen full-fit values and optional evaluation caches. Reuse identical data, model and freeze settings across fitting and uncertainty stages.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--collect-skip-diagonal` | `False` | Do not load the diagonal HessianParallel files, used for numerical where they are not created |
| `--include-per-model-lnN` | `False` | Include the lnN in the non-combined likelihood. |
| `--include-per-model-rate` | `False` | Include the rate parameters in the non-combined likelihood. |

## Behaviour and checks

Numerical parallel Hessians can omit diagonal entry files. Use `--collect-skip-diagonal` when collecting the corresponding off-diagonal-only output set, following the numerical-Hessian workflow.

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/infer.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
