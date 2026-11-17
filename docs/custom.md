---
layout: page
title: "Step: Custom"
---

Run an analysis-specific runner with user-supplied configuration.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="Custom" --custom-module="YourModule" --custom-options="name:value"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

## Inputs and outputs

**Requires:** An importable custom module/class and its own required inputs.

**Produces:** Outputs declared by the custom runner; the framework cannot infer them from the step name.

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--custom-module` | `None` | Name of custom module |
| `--custom-options` | `''` | Semi-colon separated list of options set by an equals sign to custom module |

## Behaviour and checks

The module must expose the runner class expected by the dispatch code and implement Configure, Run, Inputs and Outputs. The current dispatcher splits options on a colon (despite the equals-sign parser help). Pass options as a semicolon-separated string, for example `--custom-options="name:value;other:value"`. See the [step template](step_template.md).

## Redrawing Breit–Wigner fraction plots

For `--custom-module=top_bw_fractions`, set
`--custom-options="only_plot:True;plot:True;plot_dist:False"` to regenerate the
fraction-spline plots from the saved YAML and spline pickle files. This mode
requires no event parquet files and does not refit or overwrite the fractions.
Set `plot_dist:True` as well to regenerate the reweighted distributions and ratio
plots; these additionally require the original event parquet files and matching
selection options. All plots include the translated category label.

For binned fits with multiple channels per era, add `group_by_base_file:True`
to the custom options. The runner groups channels using
`inference.binned_fit.files.ttbar`, applies the union of their selections, and
reads each era-level base file once per fraction calculation. Spline keys come
from each base file's `bw_fractions.args.category`, so the generated locations
match the preprocessing consumer. Plot labels and output names use those era
keys. Use the same grouping option when redrawing the saved plots. The default
remains one calculation per configured category.

## Implementation

[CLI dispatch](../scripts/innfer.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).
