# Consolidating the BTM models

`configs/run/btm_full_171126_sbi.py` uses the `BTM_Full_171126` namespace.
It preserves the balanced-split smaller-range preprocessing, conditions,
dequantisation and observables. It shares the existing raw input files.

From the repository root, in the innfer environment:

```bash
# Check every required source file and write the proposed copy plan.
python3 scripts/prepare_btm_full_171126.py

# Perform the copy into data/, plots/ and models/BTM_Full_171126.
python3 scripts/prepare_btm_full_171126.py --apply
```

The selection is recorded in `configs/migration/btm_full_171126.json`:

| Models | Source | New model name |
| --- | --- | --- |
| Six density models | `_density_retrain` main saved models | Nominal, without suffix |
| 25 selected classifiers | `_collapsed_fullgrad` main best-Test models | Nominal, without suffix |
| Remaining 440 classifiers | Nominal balanced-split models | Nominal |

The full-gradient main weights are the saved best-Test checkpoints, rather than
the final 8000-update snapshots. This selection includes all 25 requested models;
it does not imply that every remaining response discrepancy has been resolved.
Epoch snapshots and other experimental model directories stay in the old run.

The script copies `PreProcess`, `LoadData`, `DataCategories`, `top_bw_fractions`,
`SimToDataFactors`, and every `InputPlot*` plot directory. It copies matching
classifier evaluation and nuisance-validation YAML summaries for the selected
models, with Full, TestInf and Val kept separate. Available classifier performance
and density-retrain performance summaries are carried across too. It does not
reuse generated samples or generator, p-value and inference plots from other models.
Nuisance plots and cached predictions can be regenerated from the promoted weights.

Pruning on the held-out nuisance metrics uses the same option as before:

```bash
--prune-from=ClassifierNuisanceVariationsTestInf
```

Copies are independent files, including dereferenced source symlinks. Embedded
run paths in text metadata are rewritten to the new namespace; numerical
Parquet, HDF5 and pickle contents are preserved. Model-weight checksums and the
complete source-to-destination plan are recorded in
`data/BTM_Full_171126/migration_manifest.json`. Its status becomes `complete`
only after copying, metadata rebasing and weight verification finish.

If the copy is interrupted, continue it with:

```bash
python3 scripts/prepare_btm_full_171126.py --apply --resume
```

An existing destination is otherwise rejected. A completed migration cannot be
overwritten or resumed, protecting subsequent work in the new run. After copying,
use `--cfg=configs/run/btm_full_171126_sbi.py` with the usual steps and no density
or classifier study suffixes. Validation using synthetic samples should be rerun
in the new namespace.

Run the complete validation workflow using the promoted nominal models:

```bash
python3 scripts/innfer.py \
  --cfg=configs/run/btm_full_171126_sbi.py \
  --step=SnakeMake \
  --snakemake-cfg=configs/snakemake/btm_validation.yaml
```

This includes classifier metrics and nuisance/response plots, density-only and
full-model Generator plots (including nuisance variations), prefit comparisons,
p-values (100 synthetic toys per hypothesis), calibration, combined Asimov fits,
and combined/per-era frozen inference closure summaries. Generator and distribution
plots show the hypothesis. The workflow reuses the copied preprocessing and
promoted model weights; it does not train or preprocess.

Fresh density metrics are stored under `DensityPerformanceMetricsValidation`,
including mass closure on `train_inf`, `test_inf` and `val` for ttbar. Classifier
pruning retains the existing `q75:1.01` threshold on TestInf nuisance metrics.
Completed up-to-date validation products are reused on subsequent launches.
