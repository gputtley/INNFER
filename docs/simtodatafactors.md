---
layout: page
title: "Step: SimToDataFactors"
---

Derive simulation-to-data yield normalisation factors for selected processes and category groups.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="SimToDataFactors"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

## Inputs and outputs

**Requires:** DataCategories data tables and nominal-yield preprocessing fragments.

**Produces:** normalisation_factors_{category}.yaml files used by yield collection.

| Location | Path pattern |
| --- | --- |
| Results | `$PREP_DATA_DIR/$CFG_NAME/SimToDataFactors{extra_output_dir_name}/normalisation_factors` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--sim-to-data-norm-groups` | `None` | The category groups to normalise simulation to data for. Comma separated category names, for groups to merge, then semi colon separated. |
| `--sim-to-data-norm-processes` | `'all'` | The processes to normalise simulation to data for. Comma separated process names, all means all processes |

## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/sim_to_data_factors.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
