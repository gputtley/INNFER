---
layout: page
title: "Running INNFER"
---

Running INNFER happens though the `scripts/innfer.py` script, this can be called by the `innfer` alias. 

You must provide two accompanying options:
- A config file parsed with the `--cfg` option (discussed [here](config.md)), or a benchmark name with the `--benchmark` option.
- The step you want to run with the `--step` option. More information about the steps are detailed further below

Therefore, an example command to run **INNFER** is shown below.
```bash
innfer --cfg="example_cfg.yaml" --step="PreProcess"
```

As commands may take some time, jobs can be parallelised and submitted to a batch system such as HTCondor, by adding `--submit="condor.yaml"`, which points to a configuration file for submission to the batch. The submission options can be altered in this yaml file. Running INNFER on a batch is highly recommended in all cases.

Example workflows that have been used in analyses are available in the workflows folder.

# Available Steps

## Table of Contents
- [MakeBenchmark](#makebenchmark)
- [LoadData](#loaddata)
- [PreProcess](#preprocess)
- [ResampleValidationForData](#resamplevalidationfordata)
- [DataCategories](#datacategories)
- [Custom](#custom)
- [InputPlotTraining](#inputplottraining)
- [InputPlotValidation](#inputplotvalidation)
- [TrainDensity](#traindensity)
- [EvaluateDensity](#evaluatedensity)
- [PlotDensity](#plotdensity)
- [SetupDensityFromBenchmark](#setupdensityfrombenchmark)
- [TrainRegression](#trainregression)
- [EvaluateRegression](#evaluateregression)
- [PlotRegression](#plotregression)
- [TrainClassifier](#trainclassifier)
- [EvaluateClassifier](#evaluateclassifier)
- [PlotClassifier](#plotclassifier)
- [MakeAsimov](#makeasimov)
- [DensityPerformanceMetrics](#densityperformancemetrics)
- [EpochPerformanceMetricsPlot](#epochperformancemetricsplot)
- [PValueSimVsSynth](#pvaluesimvssynth)
- [PValueSynthVsSynth](#pvaluesynthvssynth)
- [PValueSynthVsSynthCollect](#pvaluesynthvssynthcollect)
- [PValueDatasetComparisonPlot](#pvaluedatasetcomparisonplot)
- [HyperparameterScan](#hyperparameterscan)
- [HyperparameterScanCollect](#hyperparameterscancollect)
- [BayesianHyperparameterTuning](#bayesianhyperparametertuning)
- [Flow](#flow)
- [Generator](#generator)
- [GeneratorSummary](#generatorsummary)
- [LikelihoodDebug](#likelihooddebug)
- [InitialFit](#initialfit)
- [ApproximateUncertainty](#approximateuncertainty)
- [Hessian](#hessian)
- [HessianParallel](#hessianparallel)
- [HessianCollect](#hessiancollect)
- [HessianNumerical](#hessiannumerical)
- [Covariance](#covariance)
- [DMatrix](#dmatrix)
- [CovarianceWithDMatrix](#covariancewithdmatrix)
- [ScanPointsFromApproximate](#scanpointsfromapproximate)
- [ScanPointsFromHessian](#scanpointsfromhessian)
- [Scan](#scan)
- [ScanCollect](#scancollect)
- [ScanPlot](#scanplot)
- [MakePostFitAsimov](#makepostfitasimov)
- [PostFitPlot](#postfitplot)
- [SummaryChiSquared](#summarychisquared)
- [SummaryAllButOneCollect](#summaryallbutonecollect)
- [Summary](#summary)
- [SummaryPerVal](#summaryperval)

## MakeBenchmark
This will create a dataset and a yaml config file to run from for the benchmark scenario specified (with `--benchmark="Dim5"` for example) in the `python/worker/benchmarks.py` class.

## LoadData
Loads data to create base datasets.

## PreProcess
Prepares datasets for training, testing, and validation. Includes standardisation, train/test/validation splitting, and optional binned fit input.

## ResampleValidationForData
~  

## DataCategories
~  

## Custom
Runs a custom module specified by the `--custom-module` option. Additional options can be passed using `--custom-options`.

## InputPlotTraining
Plots training and testing datasets after preprocessing.

## InputPlotValidation
Plots validation datasets after preprocessing.

## TrainDensity
Trains density networks using the specified architecture. Supports logging with wandb and saving models per epoch.

## EvaluateDensity
~  

## PlotDensity
~  

## SetupDensityFromBenchmark
Sets up density models based on a benchmark scenario.

## TrainRegression
Trains regression networks for specific parameters using the specified architecture.

## EvaluateRegression
Evaluates regression models on validation datasets, also makes the normalisation spline.

## PlotRegression
Plots regression results of average binned value vs average regressed value.

## TrainClassifier
~  

## EvaluateClassifier
~  

## PlotClassifier
~  

## MakeAsimov
Generates Asimov datasets for validation. The number of events can be specified using `--number-of-asimov-events` or you can use the `--use-asimov-scaling` option to take the number of asimov events scaled from the predicted number of events.

## DensityPerformanceMetrics
Computes performance metrics for trained density networks. Supports metrics like loss, histograms, and multidimensional metrics.

## EpochPerformanceMetricsPlot
Plots performance metrics as a function of training epochs.

## PValueSimVsSynth
Performs a p-value dataset comparison test between simulated and synthetic datasets.

## PValueSynthVsSynth
Performs a p-value dataset comparison test between bootstrapped synthetic datasets.

## PValueSynthVsSynthCollect
Collects results from the synthetic vs. synthetic p-value tests.

## PValueDatasetComparisonPlot
Plots p-value dataset comparison results.

## HyperparameterScan
This will perform a hyperparameter scan based on a scan architecture file given by `--architecture="scan.yaml"` for example.  
If you want to track your trainings with wandb, you first must need to set up an account at [https://wandb.ai/](https://wandb.ai/). Then you must connect your terminal to your account with the command `wandb login`. You will need to enter the API given on the website. Now you can add `--use-wandb` to the command and track online.

## HyperparameterScanCollect
Collects the best-performing model from a hyperparameter scan.

## BayesianHyperparameterTuning
Performs Bayesian hyperparameter tuning using a specified architecture. Supports logging with wandb in the same way `HyperparameterScan` does.

## Flow
Visualizes the flow of the network through coupling layers.

## Generator
Uses the network as a generator to create plots comparing network outputs to input simulations. Supports 1D and 2D unrolled plots.

## GeneratorSummary
Creates summary plots comparing simulations and network-generated outputs across all unique Y values.

## LikelihoodDebug
Debugs likelihood calculations for specific parameter values. It allows you to evaluate the likelihood with the parameter values given by the `--other-input` option.

## InitialFit
Performs minimization of the negative log-likelihood to find the best fit.  
Whenever calling the infer module you can specify the `--likelihood-type`, `--data-type` and numerous other options. Supports specifying the minimization method with `--minimisation-method`.

## ApproximateUncertainty
Calculates approximate uncertainties by interpolating likelihood crossings.

## Hessian
Computes the Hessian matrix for the likelihood.

## HessianParallel
Computes the Hessian matrix in parallel for efficiency.

## HessianCollect
Collects results from parallel Hessian computations.

## HessianNumerical
Computes the Hessian matrix numerically.

## Covariance
Calculates the covariance matrix from the inverse of the Hessian.

## DMatrix
Computes the D-matrix for asymptotic corrections, as derived [here](https://arxiv.org/abs/1911.01303).

## CovarianceWithDMatrix
Computes the corrected covariance matrix using the Hessian and D-matrix.

## ScanPointsFromApproximate
Identifies points for likelihood scans based on approximate uncertainties. The number of scan points can be given by the `--number-of-scan-points` option.

## ScanPointsFromHessian
Identifies points for likelihood scans based on the Hessian matrix. The number of scan points can be given by the `--number-of-scan-points` option.

## Scan
Performs a profiled likelihood scan over identified points.

## ScanCollect
Collects results from likelihood scans.

## ScanPlot
Plots results from likelihood scans.

## MakePostFitAsimov
Generates post-fit Asimov datasets using best-fit parameter values.

## PostFitPlot
Plots distributions for best-fit results compared to the data used for fitting.

## SummaryChiSquared
Computes the chi-squared value between best-fit and truth values for validation parameters.

## SummaryAllButOneCollect
Collects results for "all-but-one" parameter summaries.

## Summary
Plots a summary of best-fit and truth values for validation parameters.

## SummaryPerVal
Plots summaries of results for each validation index.



---

Next: [Configuration File](config.md).

{% include mathjax.html %}