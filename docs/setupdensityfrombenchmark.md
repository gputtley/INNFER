---
layout: page
title: "Step: SetupDensityFromBenchmark"
---

Install a benchmark-backed density representation for comparisons with the learned likelihood.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="SetupDensityFromBenchmark"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `model_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** A benchmark run configuration and benchmark implementation.

**Produces:** Model .h5/architecture artifacts under the density model directory, as defined by the benchmark setup.

| Location | Path pattern |
| --- | --- |
| Results | `$MODELS_DIR/$CFG_NAME/{model_name}{extra_density_model_name}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/setup_density_from_benchmark.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
