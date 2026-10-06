---
layout: page
title: "P-value dataset comparison"
---

The dataset-comparison workflow asks whether a learned density reproduces simulation better than can be distinguished by the selected metric at the tested sample size. It measures density-model performance rather than validating every nuisance-ratio component of the complete likelihood.

## Run order

1. [PValueSimVsSynth](pvaluesimvssynth.md) measures simulation-versus-generated statistics and writes reference synthetic samples.
2. [PValueSynthVsSynth](pvaluesynthvssynth.md) generates alternative synthetic toys and compares them with the reference.
3. [PValueSynthVsSynthCollect](pvaluesynthvssynthcollect.md) collects toy metrics.
4. [PValueDatasetComparisonPlot](pvaluedatasetcomparisonplot.md) plots the null distribution and the observed statistic.

```bash
innfer --cfg="configs/run/your_analysis.py" \
  --step="PValueSimVsSynth,PValueSynthVsSynth,PValueSynthVsSynthCollect,PValueDatasetComparisonPlot" \
  --specific-file-name="ttbar" --specific-category="2223" \
  --pvalue-per-val-ind --number-of-toys=100
```

This sequential command runs locally. For scheduled dependencies, use `configs/snakemake/subworkflow/p_value_condor_ic.yaml` through [SnakeMake](snakemake.md), after preprocessing and training finish.

## Matching the null comparison

Keep the density model suffix, validation hypothesis, metric choices, result-directory suffixes, toy count and sample construction consistent across stages. `--pvalue-per-val-ind` separates hypotheses into directories such as `PValueSimVsSynth_val_ind_2`. Use `--specific` to select a loop index when required; `--pvalue-per-val-ind-hypothesis="bw_mass=172.5"` changes the generated hypothesis and is not a replacement for regenerating the simulation reference.

These steps request generated counts based on simulation row counts, use unit synthetic weights and retain simulation weights in the comparison path. A simulation row count is not generally its effective count when event weights vary. Metric implementations can treat signed weights differently, so examine the reported statistic and its matched toys rather than interpreting every metric as the same goodness-of-fit test.

The default multidimensional metric selection is `--density-performance-metrics-multidim=bdt`. `wasserstein` and `kmeans` select additional implementations. The BDT AUC is measured with a train/test separation in the comparison worker; finite training and sampling can shift the synthetic null mean away from exactly 0.5.

## Interpretation

The plotter uses the upper-tail estimate

$$
p = \frac{1 + \#\{T_{\mathrm{toy}}\geq T_{\mathrm{sim/synth}}\}}{N_{\mathrm{toys}}+1}.
$$

With 100 toys, the smallest value is 1/101, not zero. This resolution is separate from model quality. Recompute the null after changing the model or comparison setup; a previous null is not guaranteed to remain calibrated.

An outlying AUC identifies a distinguishable joint distribution but does not identify its cause. Follow up with marginal and multidimensional residuals, independent generated samples, transform/dequantisation checks and closure fits. Similar residuals using the same generated reference are not independent replications.

[Back to all steps](steps.md).

{% include mathjax.html %}
