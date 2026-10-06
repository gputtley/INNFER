---
layout: page
title: "Steps"
---

Each implemented CLI step has a page below with its purpose, prerequisites, outputs and options. Start with [common step options](stepoptions.md); likelihood stages also share [inference options](inferenceoptions.md).

A typical chain is LoadData → PreProcess → TrainDensity/TrainClassifier → MakeAsimov → Generator or InitialFit. Dependencies are described on each page. For batch orchestration, see [SnakeMake](snakemake.md).

## Data preparation

- [MakeBenchmark](makebenchmark.md)
- [LoadData](loaddata.md)
- [PreProcess](preprocess.md)
- [PreProcessParallelInitial](preprocessparallelinitial.md)
- [PreProcessParallelYieldsNominal](preprocessparallelyieldsnominal.md)
- [SimToDataFactors](simtodatafactors.md)
- [PreProcessParallelYieldsParameters](preprocessparallelyieldsparameters.md)
- [PreProcessParallelYieldsCollect](preprocessparallelyieldscollect.md)
- [PreProcessParallelBinnedFitInputsNominal](preprocessparallelbinnedfitinputsnominal.md)
- [PreProcessParallelBinnedFitInputsParameters](preprocessparallelbinnedfitinputsparameters.md)
- [PreProcessParallelTrainTestValSplit](preprocessparalleltraintestvalsplit.md)
- [PreProcessParallelModel](preprocessparallelmodel.md)
- [PreProcessParallelValidation](preprocessparallelvalidation.md)
- [PreProcessParallelNuisanceVariations](preprocessparallelnuisancevariations.md)
- [PreProcessParallelNuisanceDoubleVariations](preprocessparallelnuisancedoublevariations.md)
- [PreProcessParallelMerge](preprocessparallelmerge.md)
- [ResampleValidationForData](resamplevalidationfordata.md)
- [DataCategories](datacategories.md)
- [ParametersToROOT](parameterstoroot.md)
- [Custom](custom.md)
- [InputPlotTraining](inputplottraining.md)
- [InputPlotValidation](inputplotvalidation.md)
- [InputPlotNuisanceVariations](inputplotnuisancevariations.md)
- [InputPlotComparingTrainWithVariations](inputplotcomparingtrainwithvariations.md)

## Training and individual model validation

- [PlotDensityTransform](plotdensitytransform.md)
- [TrainDensity](traindensity.md)
- [EvaluateDensity](evaluatedensity.md)
- [PlotDensity](plotdensity.md)
- [SetupDensityFromBenchmark](setupdensityfrombenchmark.md)
- [TrainRegression](trainregression.md)
- [EvaluateRegression](evaluateregression.md)
- [PlotRegression](plotregression.md)
- [TrainClassifier](trainclassifier.md)
- [EvaluateClassifier](evaluateclassifier.md)
- [PlotClassifier](plotclassifier.md)
- [ClassifierPerformanceMetrics](classifierperformancemetrics.md)

## Synthetic generation and model performance

- [MakeAsimov](makeasimov.md)
- [MakeAsimovNuisanceVariations](makeasimovnuisancevariations.md)
- [MakeAsimovNuisanceDoubleVariations](makeasimovnuisancedoublevariations.md)
- [MakeAsimovForNormalisationSpline](makeasimovfornormalisationspline.md)
- [NormalisationSplineClassifier](normalisationsplineclassifier.md)
- [DensityPerformanceMetrics](densityperformancemetrics.md)
- [EpochPerformanceMetricsPlot](epochperformancemetricsplot.md)
- [PValueSimVsSynth](pvaluesimvssynth.md)
- [PValueSynthVsSynth](pvaluesynthvssynth.md)
- [PValueSynthVsSynthCollect](pvaluesynthvssynthcollect.md)
- [PValueDatasetComparisonPlot](pvaluedatasetcomparisonplot.md)
- [HyperparameterScan](hyperparameterscan.md)
- [HyperparameterScanCollect](hyperparameterscancollect.md)
- [BayesianHyperparameterTuning](bayesianhyperparametertuning.md)
- [Flow](flow.md)
- [CalibrationPlot](calibrationplot.md)
- [Generator](generator.md)
- [GeneratorSummary](generatorsummary.md)
- [GeneratorNuisanceVariations](generatornuisancevariations.md)
- [ClassifierNuisanceVariations](classifiernuisancevariations.md)
- [ValidationPerformanceMetrics](validationperformancemetrics.md)
- [PlotFactorisation](plotfactorisation.md)

## Fitting and uncertainty estimation

- [LikelihoodDebug](likelihooddebug.md)
- [InitialFit](initialfit.md)
- [BootstrapFit](bootstrapfit.md)
- [BootstrapCollect](bootstrapcollect.md)
- [BootstrapPlot](bootstrapplot.md)
- [ApproximateUncertainty](approximateuncertainty.md)
- [UncertaintyFromMinimisation](uncertaintyfromminimisation.md)
- [Hessian](hessian.md)
- [HessianParallel](hessianparallel.md)
- [HessianCollect](hessiancollect.md)
- [HessianNumerical](hessiannumerical.md)
- [HessianNumericalParallel](hessiannumericalparallel.md)
- [Covariance](covariance.md)
- [DMatrix](dmatrix.md)
- [DMatrixNumerical](dmatrixnumerical.md)
- [CovarianceWithDMatrix](covariancewithdmatrix.md)
- [Impacts](impacts.md)
- [ImpactsCollect](impactscollect.md)
- [ApproximateImpacts](approximateimpacts.md)
- [ImpactsPlot](impactsplot.md)
- [CompareConstraintsAndImpacts](compareconstraintsandimpacts.md)
- [ScanPointsFromApproximate](scanpointsfromapproximate.md)
- [ScanPointsFromHessian](scanpointsfromhessian.md)
- [ScanPointsFromInput](scanpointsfrominput.md)
- [Scan](scan.md)
- [ScanCollect](scancollect.md)
- [ScanPlot](scanplot.md)

## Post-fit predictions and summaries

- [MakePostFitAsimov](makepostfitasimov.md)
- [MakePostFitUncertaintyAsimov](makepostfituncertaintyasimov.md)
- [DistributionPlot](distributionplot.md)
- [PostFitTruthComparison](postfittruthcomparison.md)
- [SummaryChiSquared](summarychisquared.md)
- [SummaryAllButOneCollect](summaryallbutonecollect.md)
- [Summary](summary.md)
- [SummaryPerVal](summaryperval.md)
- [SummaryNuisanceVariations](summarynuisancevariations.md)

## Workflow orchestration

- [SnakeMake](snakemake.md)

## Related guides

- [Density architecture](densityarchitecture.md)
- [P-value workflow](pvalueworkflow.md)
- [Configuration](config.md)
- [Statistical methods](statistics.md)
- [Adding a step](step_template.md)

`PostFitPlot` in older documentation refers to [DistributionPlot](distributionplot.md); it is not a current CLI selector.
