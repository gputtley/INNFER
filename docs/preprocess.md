---
layout: page
title: "Step: PreProcess"
---

Build model training/testing tables and physical validation datasets, together with the metadata needed to evaluate models.

## Run

```bash
innfer --cfg="configs/run/your_analysis.py" --step="PreProcess"
```

Replace the example configuration with your analysis configuration. [Common step options](stepoptions.md) describe process/category selection, job splitting and directory suffixes.

The dispatch exposes these loop filters where applicable: `file_name`, `category`. Use `--specific="key=value;other=value"` to select a particular iteration.

## Inputs and outputs

**Requires:** LoadData base parquet files and the configured model definitions.

**Produces:** PreProcess/{process}/{category}/parameters.yaml, density/regression/classifier training tables, and validation tables.

| Location | Path pattern |
| --- | --- |
| Results | `$PREP_DATA_DIR/$CFG_NAME/PreProcess/{process}/{category}` |

Path placeholders identify the process, category, model or optional suffix for each loop iteration; see [path conventions](stepoptions.md#directories).

## Step options

Defaults below are CLI defaults; architecture and run-configuration values are separate.

| Option | Default | Purpose |
| --- | --- | --- |
| `--number-of-shuffles` | `20` | The number of times to loop through the dataset when shuffling in preprocess |

## Dataset construction

The monolithic step runs per process/category. Base events are split before creating parameter variations or repeated copies. Training/testing datasets are constructed for each model; physical inference datasets are built separately at configured validation hypotheses.

| Dataset | Purpose |
| --- | --- |
| `density/X_train.parquet`, `Y_train.parquet`, `wt_train.parquet` | Transformed density training features, condition labels and weights. Equivalent `test` tables are used for evaluation/checkpoint selection. |
| `classifier/{parameter}/X_train.parquet`, `y_train.parquet`, `wt_train.parquet` | Classifier features including its parameter, binary varied/reference targets and balanced training weights. |
| `regression/{parameter}/X_train.parquet`, `y_train.parquet`, `wt_train.parquet` | Regression inputs, weight-variation targets and weights. |
| `val_ind_{index}/X_{split}.parquet`, `Y_{split}.parquet`, `wt_{split}.parquet` | Physical validation observables, truth labels and yield-normalised weights, for splits such as `val`, `test_inf`, `train_inf` and `full`. |
| `{nuisance}_{up_or_down}/...` | Physical nominal-hypothesis nuisance variations. |
| `parameters.yaml` | Yields, column ordering, training transforms, ranges, effective event counts and configured binned-fit information. |

Configured extra columns are written to `Extra` tables. Density model splitting can introduce `density/split_{index}` directories. The model-specific file locations in metadata are authoritative.

### Splitting

`preprocess.train_test_val_split` is a colon-separated train/test/validation fraction string such as `0.8:0.1:0.1`. The fractions must describe the intended partition before repeated training copies. `drop_from_training` can reserve selected values for validation. `full` combines the underlying partitions; it is not an independent validation sample.

`preprocess.stratify_to` accepts either a column name or a mapping for joint stratification:

```python
config["preprocess"]["stratify_to"] = {
    "year_ind": None,
    "source": {"expression": "sim_mass", "bins": None, "optional": True},
    "mass": {"expression": "CombinedSubJets_mass", "bins": 8},
    "weight_magnitude": {"expression": "abs(weight)", "bins": 4},
    "weight_sign": {"expression": "weight > 0", "bins": None},
}
```

Columns/expressions must be available in the loaded base tables. `bins: None` preserves discrete classes; continuous values are quantile-binned. Sparse joint strata are merged to make the two-stage split feasible. Stratification acts within streamed preprocessing batches and improves balance of the chosen summaries; it cannot guarantee that all weighted distributions or fitted results agree between splits.

### Variations and training weights

Model definitions specify feature/weight shifts, condition sampling and `n_copies`. Copies reuse base events at different conditions; they do not add independent simulation information. Density training includes yield flattening, configured shift reweighting and selected-region normalisation before fitting transformations. Classifiers build both varied and reference classes and balance their weights.

The pipeline's shift-reweighting step uses the configured shift distribution as the target. A discontinuity or empty support in the underlying simulation cannot be repaired merely by increasing the number of copies or normalising the total yield.

### Dequantisation

```python
config["preprocess"]["dequantisation"] = {
    "SubJet2_btagDeepB": {
        "dtype": "float16", "bounds": [0.0, 1.0], "seed": 42,
    },
}
```

This adds uniform noise within each floating-point rounding cell, clipped to the optional physical bounds. It is not Gaussian smearing. Input values must be finite and exactly representable on the selected `float16` or `float32` grid; otherwise the transform raises an error.

Dequantisation is applied after selections/variations and before model transformations to model datasets, validation datasets and nuisance variations. [DataCategories](datacategories.md) uses the same configuration for observed data. Missing columns are skipped. The older `density_dequantisation` key remains a fallback; `dequantisation` takes precedence.

Random streams use the configured seed plus a dataset context. Results are reproducible for the same context/input order, but separately constructed datasets need not receive identical noise for a shared raw event. Do not reapply the transform to already dequantised inputs.

### Transforms and metadata

Model train/test tables are transformed during preprocessing. Validation and observed tables retain physical observables; density evaluation applies the saved training transformations. The spline-to-Gaussian transform is selected by `density_pretransform_to_gaussian` and optionally `density_pretransform_to_gaussian_columns`. Its parameters and standardisation must match the checkpoint used downstream.

Transformation parameters are fitted from training data, not independently from validation. The initial physical and transformed ranges are also stored. Generated min/max filtering restricts sampled support; it does not recover probability that a learned flow assigns outside the accepted range.

A provided `preprocess.standardisation` mapping can preserve training means/stds when deliberately reusing a model. Changing those numbers after training changes the model's physical interpretation. Changing dequantisation, feature definitions or fitted transforms generally requires rebuilding the model data and retraining unless the previous training representation is demonstrably unchanged.

PCA-whitening helper code exists, but the current Run path has its whitening call commented out. Do not assume setting a whitening key enables a transform that is not executed.

### Validation weights and statistics

Validation samples are normalised to the predicted yield from the saved yield metadata. Their effective counts are stored as

$$
N_{\mathrm{eff}} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

Binned validation yields are also saved when the binned-fit configuration requests them. The `test` split used to choose a training checkpoint differs from the physical `test_inf` sample used for inference closure.

### Shuffling

Preprocessing shuffles the final train/test tables. `--number-of-shuffles` defaults to **20** in the current CLI. This is separate from the optional fresh buffered shuffling during each training epoch; see [Density architecture](densityarchitecture.md#buffered-epoch-shuffling).

## Parallel preprocessing

The fine-grained workflow follows these dependencies:

```text
LoadData → yields nominal/parameters → yields collect → base split
                                                  → model writers
                                                  → validation writers
                                                  → nuisance-variation writers
model + validation + initial metadata → merge → training/validation consumers
```

Model and validation writers require the base splits as well as initial yields. If simulation-to-data normalisation is used, DataCategories and nominal yields feed SimToDataFactors before yield collection. Binned-fit fragments have their own nominal and parameter writers.

[PreProcessParallelInitial](preprocessparallelinitial.md) combines initial phases; it is an alternative to scheduling overlapping fine-grained phases. The BTM example uses the separate yield and split phases in `configs/snakemake/subworkflow/btm_preprocess_with_bw_condor_ic.yaml`.

[PreProcessParallelMerge](preprocessparallelmerge.md) combines selected YAML fragments using `--preprocess-merge` (default `initial,model,validation`). Each writer completes its own parquet collection. Merge does not turn unfinished numbered shards into complete training tables.

When rerunning only validation, merge its updated fragments with the existing model fragments. The model fragment retains the authoritative density transformations. A stale fragment alone is not proof that a failed rerun produced its data: check the final parquet files and producer logs.


## Implementation

[CLI dispatch](../scripts/innfer.py), [Runner](../python/runner/preprocess.py). Runner `Inputs()` and `Outputs()` declare the files used to construct the Snakemake dependency graph.

[Back to all steps](steps.md).

{% include mathjax.html %}
