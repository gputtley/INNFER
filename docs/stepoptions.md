---
layout: page
title: "Common step options"
---

Every step is dispatched by `scripts/innfer.py`. Source `env.sh` first, then use the `innfer` alias or `python3 scripts/innfer.py`. Run configurations can be YAML files or Python files exposing a `config` dictionary.

```bash
source env.sh
innfer --cfg="configs/run/your_analysis.py" --step="TrainDensity" \
  --specific-file-name="ttbar" --specific-category="2223"
```

## Selecting work

| Option | Behaviour |
| --- | --- |
| `--step` | Exact, case-sensitive step name. Comma-separated steps run in sequence locally. |
| `--cfg` | Analysis configuration. Its `name` determines the output namespace. |
| `--benchmark` | Benchmark implementation instead of an analysis configuration. |
| `--specific-file-name` | Comma-separated process/model-file names for steps using model-file selection; `combined` is available in combined inference/generation loops. This does not filter the LoadData base-file loop. |
| `--specific-category` | Comma-separated category names for steps that have category loops. |
| `--specific` | Filter loop keys, such as `file_name=ttbar;category=run2;model_type=classifier_models;parameter_name=AbsoluteScale`. Quote semicolons in the shell. |
| `--skip-non-density` | In steps supporting this flag, omit validation hypotheses outside the density-only validation subset. |
| `--specific-combined-default-val` | Select the default combined validation point in supporting loops. Data fits also restrict the validation loop to defaults. |
| `--loop-over-nuisances`, `--loop-over-rates`, `--loop-over-lnN` | Include those parameter classes in scans/uncertainty loops that normally select POIs. |
| `--loop-over-only-val-parameters` | Restrict supported parameter loops to parameters varied by the validation setup. |
| `--validation-loop-over-nuisance-variations` | Use nuisance-variation hypotheses instead of the ordinary validation loop in supporting inference steps. |

Loop-key names differ by step. The dispatched job command and the `loop` dictionary in [scripts/innfer.py](../scripts/innfer.py) identify valid filters. `val_ind` is an index into the configured validation loop, not a mass value.

## Directories

Each root is extended by the configuration's `name`. In the path patterns used throughout these docs, `$CFG_NAME` means `config['name']`; it need not be an exported environment variable.

| Root | Typical default | Contents |
| --- | --- | --- |
| `$PREP_DATA_DIR/$CFG_NAME` | `data/{configuration_name}` | Loaded and preprocessed datasets, observed-data categories. |
| `$EVAL_DATA_DIR/$CFG_NAME` | `data/{configuration_name}` | Metrics, synthetic datasets, fit and scan results. |
| `$MODELS_DIR/$CFG_NAME` | `models/{configuration_name}` | Model weights, architectures and normalisation splines. |
| `$PLOTS_DIR/$CFG_NAME` | `plots/{configuration_name}` | Diagnostic and result plots. |
| `$JOBS_DIR/$CFG_NAME` | `jobs/{configuration_name}` | Generated scripts, logs and the Snakefile. |

Common model names are `density_{process}_{category}`, `classifier_{process}_{parameter}_{category}` and `regression_{process}_{parameter}_{category}`. Split-density configurations can introduce additional model directories. `{process}` denotes a key in the run configuration's `models`, while `{base_file_name}` denotes a key in `files`. Braced names in documentation paths are placeholders, not literal filenames.

### Suffixes

`--extra-output-dir-name=Trial` appends `Trial` to the step directory name, for example `DensityPerformanceMetricsTrial`. `--extra-input-dir-name=Trial` selects the corresponding upstream result directories. `--extra-dir-name=Trial` sets both. Suffixes are concatenated without inserting a separator; include `_` yourself if desired.

Model selection has separate flags: `--extra-density-model-name`, `--extra-classifier-model-name` and `--extra-regression-model-name`. These select model-directory variants in the steps that support them. They do not imply a matching result-directory suffix.

Synthetic inputs have their own `--extra-asimov-input-dir-name` and `--extra-postfit-asimov-input-dir-name`. Keep them consistent with the generating stage.

`--density-correction` opts Asimov generation, density PValue comparisons and unbinned likelihood evaluations into the classifier ratio trained by `DensityCorrectionWithClassifier`. It defaults to false. Use `--extra-density-correction-input-dir-name` to select correction artifacts trained with an output-directory suffix; density and classifier model suffixes select the checkpoint variant. The correction must match the density checkpoint and its saved transforms. Fits, scans, Hessians, DMatrix, calibration and density inference-closure metrics use the normalized corrected density. Classifier-model PValue comparisons are unaffected. Density tuning metrics also receive the option, but a correction trained for a different trial checkpoint is rejected. See [inference options](inferenceoptions.md) for normalization and derivative settings.

`--add-specific-category-to-dir-name` appends `_{specific_category}` to the input/output directory suffixes and the Asimov input suffix. Use it explicitly when a fit directory otherwise combines categories. In nested workflows, inherited options apply to descendants; sibling workflow overrides do not leak into one another.

## Batch submission and dependencies

`--submit=configs/submit/condor_4_cpus.yaml` submits the selected work using that batch configuration. `--points-per-job` groups loop iterations into one job; its default is 1. `--dry-run` prepares batch submission without submitting it. For a dependency graph and restartable workflow, use [SnakeMake](snakemake.md).

Each runner declares `Inputs()` and `Outputs()`. Missing inputs prevent execution or workflow construction. Running a training-only subworkflow does not automatically add preprocessing jobs: either produce those inputs first or include their producer subworkflow.

`--add-inputs` and `--add-outputs` add comma-separated dependency paths. This is useful for external correction files referenced by configuration callbacks. `--ignore-inputs-and-outputs` bypasses existence checks; it does not create missing datasets.

A numbered shard such as `X_train_1.parquet` is not interchangeable with the final `X_train.parquet`. Successful completion requires the final outputs declared by the producer. Old parameter fragments can survive an interrupted rerun and are not proof that its datasets finished.

## Discovering options

`innfer --help` lists parser options and defaults. The [steps index](steps.md) follows the implemented dispatch branches. The separate registry used by `--list-steps` and `--describe` may lag the implemented step set.

[Back to all steps](steps.md).
