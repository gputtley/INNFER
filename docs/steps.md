---
layout: page
title: "Steps"
---

## Available Steps

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

This step is purely for testing of the repository from a dataset with a known probability distribution function (PDF). If you are running using a configuration file, this step can be ignored.

The step itself builds the input dataset from the known PDF, and the yaml configuration file setup to correctly run **INNFER**. To run this step, the known density is provided from the `python/worker/benchmarks.py` class. This is done with the `--benchmark=Dim5` option, for example.

The benchmark scenarios are simple examples set up with a number of observables conditional on a set of parameters. Later when performing inference, the results using true PDF of the benchmark, stored in the class, can be used to compare to the learned PDF.

The benchmark scenarios that are set up are:
- **Dim1Gaussian**: A single Gaussian observable conditional on a single parameter which is the mean and which scales the width of the Gaussian.
- **Dim1GaussianWithExpBkg**: The PDF from **Dim1Gaussian** representing a 'signal', stacked on top of a fixed exponentially falling 'background'.
- **Dim1GaussianWithExpBkgVaryingYield**: Equivalent to **Dim1GaussianWithExpBkg**, except separate density models are formed for the 'signal' and 'background'. They are combined at the time of inference with a freely floating rate parameter on the 'signal' yield.
- **Dim2**: Two observables, a Gaussian and a chi squared distribution, conditional on one parameter.
- **Dim5**: Five observables, a Gaussian, chi squared, exponential, beta and Weibull distribution, conditional on one parameter.

When running the remaining steps from a benchmark you can either continue to parse `--benchmark=Dim5` instead of the configuration file, or the created configuration file with `--cfg=Benchmark_Dim5.yaml`.

## LoadData

This step is typically the first step when running **INNFER** (assuming you are not working from a benchmark). It creates the 
base datasets, defined using the "files" input in the configuration file, from which all variations (both weight and feature changing) are calculated later. 

It loads in the input datasets and adds the extra columns specified in the configuration file. There are additional options to add summed columns and remove negative weights. It will also reduce the number of stored columns to the minimum, so that the next PreProcess steps runs more efficiently. It will output one parquet dataset per "files" key in the configuration file.

## PreProcess

PreProcess is the largest of data preparation steps in **INNFER** repository. There are a number of techniques involved but overall the purpose of this step is to create datasets ready for training and testing density, regression and classifier models to use in building the likelihood and to produce validation datasets to validate the performance of the learned likelihood. It also creates a parameters yaml file which contains various information about the datasets including the names of the columns, the standardisation parameters, the information regarding the yields so we can produce a prediction for the total yield in the likelihood, and many other important features.

The methods used during the PreProcess step are:

- **Getting the yields**: 

It gets the nominal yield by calculating the sum of the weights at the default values of whichever base file is parsed as the "yield" for the particular "models" input in the configuration file. It also builds that dataset to get the 1 sigma variations of the nuisance parameters and save the sum of weights to find the relative yield effects of each nuisance, so this can be factored out and included as a log normal (lnN) nuisance parameter in the likelihood.

- **Get the binned fit inputs**: 

This method is only run if the "binned_fit_input" is defined in the configuration file. As **INNFER** is built for unbinned likelihood fits, this is only setup for a cross check of the unbinned results against the binned results. If you want to use a rigorous statistical framework for binned analysis, we recommend using [Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/). If you still wish to use binned fits in **INNFER**, this will calculate the yields and lnN effects in each bin and write this to the parameters file, for later use when building the binned likelihood.

- **Train, test and validation splitting**: 
- **Model variation**: 
- **Validation normalisation**:
- **Get validation effective events**:
- **Get validation binned histograms**:
- **Flatten by yields**:
- **Normalise in**:
- **Data Standardisation**:
- **Classifier class balancing**:
- **Shuffle training and testing datasets**:
- **Make parameters file**:

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


<br>

---

Next: [Configuration File](config.md).

{% include mathjax.html %}