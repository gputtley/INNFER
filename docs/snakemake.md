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

## Dependencies and external files

Include all producer stages needed by your target workflow. A training-only subworkflow assumes final preprocessed parquet files already exist. An existing parameters.yaml or an intermediate shard does not satisfy a missing X_train.parquet.

External callbacks can reference correction files outside the runner's declared inputs. Add explicit dependencies when needed:

```yaml
run_options:
  add_inputs: "$PREP_DATA_DIR/$CFG_NAME/top_bw_fractions/top_bw_fraction_locations.yaml"
```

`add_inputs` and `add_outputs` accept comma-separated paths. `$CFG_NAME` is substituted with the run configuration name by workflow generation. Track referenced payload files as well as an index file if changes to those payloads must trigger rebuilding.

## Inspecting and restarting

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

[Back to all steps](steps.md).
