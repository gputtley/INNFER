---
layout: page
title: "Step: PreProcessParallelYieldsCollect"
---

Merge nominal and varied yields into the initial metadata used downstream.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PreProcessParallelYieldsCollect"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** Nominal/parameter yield fragments; SimToDataFactors files if sim-to-data-norm is enabled.

**Produces:** parameters_initial.yaml.

| Location | Path pattern |
| --- | --- |
| Results | `$PREP_DATA_DIR/$CFG_NAME/PreProcess/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

See [PreProcess](preprocess.md) for the data transformations and [parallel preprocessing](preprocess.md#parallel-preprocessing) for phase ordering.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--number-of-shuffles` | `20` | The number of times to loop through the dataset when shuffling in preprocess |
| `--sim-to-data-norm` | `False` | Normalise simulation to data in PreProcessParallelYieldsNominal step |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/preprocess.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
