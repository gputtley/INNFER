---
layout: page
title: "Running INNFER"
---

Start each session from the repository root:

```bash
source env.sh
```

The `innfer` alias invokes `scripts/innfer.py`. Select an analysis configuration with `--cfg` (YAML or a Python file exposing `config`) and an implemented step with `--step`:

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PreProcess"
```

For benchmarks, MakeBenchmark accepts `--benchmark` and writes the corresponding run configuration. Use that configuration for the following steps; see [MakeBenchmark](makebenchmark.md).

## A local workflow

```bash
innfer --cfg="configs/run/your_analysis.py" --step="LoadData,PreProcess,TrainDensity"
innfer --cfg="configs/run/your_analysis.py" --step="DensityPerformanceMetrics" \
  --specific-file-name="ttbar" --specific-category="2223"
```

The steps in a comma-separated command run sequentially locally. They require a configuration defining the requested models and sources. Classifier/regression models and their normalisation stages must also be produced when the combined likelihood needs them.

Use [common step options](stepoptions.md) to select individual loop iterations, configure suffixes and submit batches. `--specific` filters keys such as `model_name` or `file_name` depending on the step; inspect the dispatched command or CLI loop for the appropriate names.

## Output roots

```bash
export PREP_DATA_DIR="./data"
export EVAL_DATA_DIR="./data"
export MODELS_DIR="./models"
export PLOTS_DIR="./plots"
export JOBS_DIR="./jobs"
```

The configuration name is appended to each root. Defaults are set by the environment setup. `EVENTS_PER_BATCH` and `EVENTS_PER_BATCH_FOR_PREPROCESS` control streaming worker batch sizes; architecture `batch_size` controls optimiser batches. They serve different purposes.

## Scheduled workflows

Use [SnakeMake](snakemake.md) for dependency-aware execution. Submitting comma-separated steps as independent batch jobs is not a substitute for a dependency graph. A subset workflow expects its omitted upstream outputs to exist.

Useful guides:

- [All steps](steps.md): supported selectors and their input/output requirements.
- [Configuration](config.md): datasets, model definitions and validation hypotheses.
- [Density architecture](densityarchitecture.md): flow and training options.
- [Inference options](inferenceoptions.md): fitting, freezing and caches.
- [P-value workflow](pvalueworkflow.md): matching statistics and null toys.

Next: [Steps](steps.md).
