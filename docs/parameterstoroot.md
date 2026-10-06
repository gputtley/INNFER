---
layout: page
title: "Step: ParametersToROOT"
---

Export the prepared binned yields and nuisance variations as ROOT histograms.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="ParametersToROOT"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

## Inputs and outputs

**Requires:** PreProcess parameters.yaml files containing binned-fit metadata.

**Produces:** ParametersToROOT/datacards.root containing the prepared binned histograms.

| Location | Path pattern |
| --- | --- |
| Results | `$EVAL_DATA_DIR/$CFG_NAME/ParametersToROOT/datacards.root` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/parameters_to_root.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
