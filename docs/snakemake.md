---
layout: page
title: "Step: SnakeMake"
---

SnakeMake expands a workflow YAML into job scripts and a dependency graph, then runs Snakemake locally or through the HTCondor profile. Prerequisites are declared by each runner's `Inputs()` and `Outputs()`; YAML ordering alone does not impose execution order.

## Run

```bash
source env.sh
innfer --cfg="configs/run/your_analysis.py" --step="SnakeMake" \
  --snakemake-cfg="configs/snakemake/your_workflow.yaml"
```

The generated file is `$JOBS_DIR/$CFG_NAME/innfer_SnakeMake.txt`. Generation replaces the previous file in that namespace. Use a persistent terminal session for long workflows.

Submitted job files use `$JOBS_DIR/$CFG_NAME/<step>/<group>/`, for both ordinary
and container jobs. Each numbered group contains up to 500 jobs: `00000` holds
indices 0--499, `00001` holds 500--999, and so on. Scripts, application logs,
direct-submission files and container manifests for a job stay together.
Descriptive job names use stable `named_*` subdirectories. This keeps large
steps within AFS directory-entry limits. The Snakefile stays at the path above
and rule names do not change. Regenerate to use the new layout; existing
running workflows keep their recorded job paths. If an old flat jobs directory
is already full, select a fresh shared `JOBS_DIR` before regenerating.

## Workflow format

```yaml
- workflow: configs/snakemake/subworkflow/btm_preprocess_with_bw_condor_ic.yaml
  run_options:
    specific_category: "2223"

- step: TrainDensity
  submit: configs/submit/condor_gpu_long.yaml
  run_options:
    specific_category: "2223"
    specific_file_name: "ttbar"
    density_architecture: configs/architecture/density_default.yaml
    disable_tqdm: True
```

A `workflow` entry includes another YAML list recursively. A `step` entry names an implemented CLI step. `run_options` use argparse attribute names with underscores, without leading dashes. Boolean options use YAML booleans. The `submit` entry points to a batch configuration.

Outer workflow options are inherited by descendants, and a child's own value overrides the inherited value. Child overrides do not affect later siblings. Command-line options provide the initial defaults unless a workflow overrides them. To alter architecture-internal options such as buffered shuffling, edit/select the architecture rather than adding an unsupported CLI option to run_options.

Set `container_image: images/innfer-20261118.sif` in `run_options`, or pass `--container-image` to the SnakeMake command, to run submitted jobs with Apptainer. A child can use `container_image: null` to disable it. The image and config archive become rule inputs; GPU requests and retry resources retain their existing definitions. See [container submission](container.md#submit-jobs-with-an-image).

An explicit `{extra_extra_dir_name}` placeholder in a workflow's `other_input`
is replaced during workflow expansion using that step's resolved suffix (or an
empty string). For example,
`Stat. Only:ScanCollectAsimovStatOnly{extra_extra_dir_name}:scan_collect` points
to the `_run2` statistical-only fit when the included fit workflow inherits
`extra_extra_dir_name: _run2`. This keeps comparison plots in the same fit
namespace without placing category wildcard characters in directory names.

## Dependencies and external files

Include all producer stages needed by your target workflow. A training-only subworkflow assumes final preprocessed parquet files already exist. An existing parameters.yaml or an intermediate shard does not satisfy a missing X_train.parquet.

External callbacks can reference correction files outside the runner's declared inputs. Add explicit dependencies when needed:

```yaml
run_options:
  add_inputs: "$PREP_DATA_DIR/$CFG_NAME/top_bw_fractions/top_bw_fraction_locations.yaml"
```

`add_inputs` and `add_outputs` accept comma-separated paths. `$CFG_NAME` is substituted with the run configuration name by workflow generation. Track referenced payload files as well as an index file if changes to those payloads must trigger rebuilding.

## Inspecting and restarting

Set `retry_runtime` in minutes beside `submit` to request more time after a failed attempt:

```yaml
- step: BayesianHyperparameterTuning
  submit: configs/submit/condor_2_cpus_avx.yaml
  retry_runtime: 720
  run_options:
    model_type: classifier
```

The first attempt retains the submission YAML settings. Subsequent attempts override only Condor's `+MaxRuntime`, here to twelve hours; CPU, memory and requirements stay the same. Without this field, retries retain their original runtime. The HTCondor profile's `grid-submit.py` must map the positive `resources.retry_runtime` value to seconds in `+MaxRuntime`; the current user profile has this mapping. Retry counts still come from the profile. A `workflow` include can provide an inherited value, with individual steps overriding it or using `retry_runtime: null` to disable it. `--retry-runtime` supplies a global default.

Regenerate and restart the workflow to apply the new declarations. Already running workflows and `--snakemake-use-file` keep their existing declarations. The classifier tuning workflow now requests ten-hour retries (`retry_runtime: 600`) after its original three-hour attempts.

An optional `retry_memory: 16000` beside `submit` requests 16,000 MB on retries by overriding only Condor's `request_memory`. First attempts retain their original memory request. It can be used independently or together with `retry_runtime`, inherits through workflow includes, and accepts `null` to disable an inherited value. `--retry-memory` supplies a global default. Values must be positive integers in MB; the current Condor profile maps `resources.retry_memory` to `request_memory`. No workflows enable this option by default.

| Flag | Behaviour |
| --- | --- |
| `--snakemake-dry-run` | Generate scripts/Snakefile without invoking Snakemake. It does not itself run Snakemake's DAG dry run. |
| `--snakemake-use-file` | Reuse the already generated Snakefile; does not incorporate later configuration/code changes into its declarations. |
| `--snakemake-local` | Use the local executor instead of the `htcondor` profile. |
| `--snakemake-force-local` | Force generated jobs into the local-execution path in supporting generation logic. |
| `--snakemake-force` | Request `--forceall` when invoking Snakemake. |
| `--snakemake-rerun-incomplete` | The current wrapper constructs this flag but overwrites it before the final execution command; pass `--rerun-incomplete` directly to Snakemake when needed. |
| `--snakemake-directory` | Set the working directory passed to Snakemake. |

For an actual DAG dry run after generating the file:

```bash
snakemake --dry-run --cores 1 -s jobs/YourConfigName/innfer_SnakeMake.txt
```

Replace the path with the generated namespace. This checks missing producer dependencies without launching training. Regenerate after changing workflow options or dependency declarations; use-file intentionally preserves the old graph.

## BTM example

`configs/snakemake/btm.yaml` includes upstream data/BW preparation, the per-category workflow for run2, 2223 and 24, and the post-category workflow when those entries are uncommented. Observed-data fitting is a separate included workflow and only runs when enabled. Inspect the current YAML before launching: commented entries are excluded.

`btm_per_category.yaml` covers preprocessing, prefit checks, training/tuning, generated samples, closure, p-values and calibration. For a faster single-category experiment, select the appropriate subworkflow with `--specific-category=2223` and make sure its prerequisite data/models already exist.

## BTM binned example

```bash
python3 scripts/innfer.py --cfg=configs/run/btm_full_171126_binned.py \
  --step=SnakeMake --snakemake-cfg=configs/snakemake/btm_binned.yaml
```

This workflow includes LoadData, grouped era-level BW fractions, observed-data
categories, yield normalization, binned templates, validation and prefit plots.
Normalization combines all accepted channels within each era. It then runs the
existing combined Asimov fit and the Run2, 2223 and 24 fits, including their
statistical-only comparisons and postfit plots. The era filters are `run2_*`,
`2223_*`, `24_*`, with output suffixes `_run2`, `_2223`, `_24`.

The BW grouping option is enabled in the binned-specific loading subworkflow;
ordinary SBI BW preparation keeps its existing per-category behavior. The
modelling-template mass-dependence override remains enabled for
`CR1,CR2,ERDOn,ue,hdamp`. Binned preprocessing writes the normal preprocessing
and binned-fit metadata used by inference; no density/classifier training is
included. Without `--snakemake-force`, completed up-to-date products are reused.

[Back to all steps](steps.md).
