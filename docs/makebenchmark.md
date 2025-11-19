---
layout: page
title: "Step: MakeBenchmark"
---

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