---
layout: page
title: "Step: MakeBenchmark"
---

Generate a benchmark dataset and its run configuration from an importable benchmark implementation.

## Run

```bash
innfer --benchmark="Dim5" --step="MakeBenchmark"
```

The example selects the Dim5 benchmark. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

## Inputs and outputs

**Requires:** A benchmark module selected with --benchmark; no preprocessed dataset is required.

**Produces:** A configs/run/Benchmark_{benchmark}.yaml configuration and datasets created by the benchmark implementation.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--extra-job-name` | `''` | Add extra name to the submitted job |

## Available benchmarks

Implementations in `python/worker/benchmarks` include Dim1Gaussian, Dim1GaussianWithExpBkg, Dim1GaussianWithExpBkgVaryingYield, Dim2 and Dim5. Their known densities support comparisons with learned likelihoods. For subsequent steps, use the generated configuration, for example:

```bash
innfer --cfg="configs/run/Benchmark_Dim5.yaml" --step="LoadData,PreProcess"
```

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/make_benchmark.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
