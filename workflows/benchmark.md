# Benchmark

## Set benchmark file

Firstly, lets set the benchmark file.

```
benchmark="Dim5"
```

## SnakeMake running

The full workflow can be run with snakemake. This will utilise the inputs and outputs of each job to submit each job only when the inputs are available. To do this, open up a new terminal (recommend tmux or screen) and run this following command with the relevant snakemake submission configuration.

```
innfer --step="SnakeMake" --benchmark="${benchmark}" --snakemake-cfg="benchmark_condor_ic.yaml"
```

## Step-by-step running

These instructions are the step-by-step commands to be ran if you are not using the snakemake workflow. These can be particularly useful for debugging. Note each step can be submitted to a batch cluster by using the --submit="condor.yaml" for example. This will split the jobs as finely as possible. If you want to group these together you can use the --points-per-job option.

### PreProcess datasets

A few steps are needed preprocess the relevant training, testing and validation. Firstly, the benchmark inputs and running configuration file are made with the following command.

```
innfer --step="MakeBenchmark" --benchmark="${benchmark}"
```

Next, the base files are created.

```
innfer --step="LoadData" --benchmark="${benchmark}"
```

The training, testing and validation files, as well as the parameter files are created with this PreProcess step.

```
innfer --step="PreProcess" --benchmark="${benchmark}"
```

### Visualising datasets

To visualise these datasets there are two modules, one for training and testing datasets, and one for validation datasets.

```
innfer --step="InputPlotTraining" --benchmark="${benchmark}"
```
```
innfer --step="InputPlotValidation" --benchmark="${benchmark}"
```

### Training networks

The next step is to train the density network. Here we use a BayesFlow density model. If you are going to submit this to a batch, it is recommended you add the --disable-tqdm option.

```
innfer --step="TrainDensity" --benchmark="${benchmark}" --density-architecture="density_default.yaml"
```

### Density performance metrics

To test the performance of the density model, there is a module to get some performance metrics of the learned density. This is also utilised in the HyperparameterScan and BayesianHyperparameterTuning steps. To choose what performance metrics to get you can parse the --density-performance-metrics option, providing a comma separated list of any combination of loss, histogram, multidim, and inference.

```
innfer --step="DensityPerformanceMetrics" --benchmark="${benchmark}"
```

### Make asimov datasets

To create an asimov dataset parquet file the following command is available. This is utilised in the generator code and the inference code when using the option --data-type="asimov". There are two options to choose the number of events. The first is to parse --number-of-asimov-events=1000000 (default), this will create this many asimov events for each model and validation point. The second is --use-asimov-scaling=10, this will create 10 times the asimov events the the expected yield for that model.

```
innfer --step="MakeAsimov" --benchmark="${benchmark}" --use-asimov-scaling=10
```

### Generator plots

To test the distribution of the asimov events versus the validation datasets, you can plot the distribution of each with respect to the input variables with the following command. You can also draw a 2d unrolled plot with the --plot-2d-unrolled, or the standardised versions of the variables with --plot-transformed.

```
innfer --step="Generator" --benchmark="${benchmark}"
```

To see a summary of all validation points on the same plot, you can use the following command. There is also the --val-inds option to choose the specific validation indices you want on the plot.

```
innfer --step="GeneratorSummary" --benchmark="${benchmark}"
```

### p-value plots

These steps are to test whether you can separate the learned and sampled synthetic (asimov) dataset from the simulated validation datasets. This is done in the context of a p-value. Firstly, performance metrics are found to attempt to separate the simulated and synthetic dataset. 

```
innfer --step="PValueSimVsSynth" --benchmark="${benchmark}"
```

However, these metrics have no understanding on their own. Therefore, we build up a dataset of these metrics attempting to separate two randomly sampled synthetic dataset (with the same stats as the simulated sample). The number of toys thrown can be parsed to this function. 

```
innfer --step="PValueSynthVsSynth" --benchmark="${benchmark}" --number-of-toys=100
```

These results will need to be collected into one file.

```
innfer --step="PValueSynthVsSynthCollect" --benchmark="${benchmark}" --number-of-toys=100
```

Finally, we can plot where the simulated verses synthetic metrics fall in the synthetic versus synthetic distribution and calculate a p-value.

```
innfer --step="PValueDatasetComparisonPlot" --benchmark="${benchmark}"
```

### Inference closure test

Now, we test the performance of the learned density on the validation datasets. This requires a number of steps. First of which is an initial minimisation of the likelihood. There are a number of options to consider here. The --likelihood-type can be chosen to be unbinned or unbinned_extended depending on whether you want to include the total event count as a Poisson term. There are also binned fit options, however, this requires additional configuration inputs during preprocessing which will not be documented here. The --minimisation-method can also be chosen, typically best to perform this with minuit. The --freeze option will freeze the parameter of key to value provided. There is also an option to not use the regression models. These are --only-density to skip the regression models and --skip-non-density to skip the validation indices that would require the regression models. There are other options beyond this that can also be parsed.

```
innfer --step="InitialFit" --benchmark="${benchmark}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

The second step is to calculate the Hessian matrix. There are a few ways to do this in the code, the way shown below is to parallelise the calculation (useful for batch submission) as much as possible and collect. This uses the gradients straight from the models. There is also the --step="Hessian" option to do this without parallelising. The other choice is to calculate this numerically and can be done with the --step="HessianNumerical" option.

```
innfer --step="HessianParallel" --benchmark="${benchmark}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

```
innfer --step="HessianCollect" --benchmark="${benchmark}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

The next steps are to correct for the true statistical uncertainty of the validation dataset, rather than the statistical uncertainty of the predicted yield. We calculate the D Matrix using the gradients directly from the model. 

```
innfer --step="DMatrix" --benchmark="${benchmark}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

The D Matrix corrected covariance matrix and the intervals are then calculated with the following command.

```
innfer --step="CovarianceWithDMatrix" --benchmark="${benchmark}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

## Inference with the true density

To compare the results of the inference closure test to what you would have got with the known density, you can run the following commands. Firstly, we need to setup the known density as a 'model'.

```
innfer --step="SetupDensityFromBenchmark" --benchmark="${benchmark}" --density-architecture="Benchmark" --extra-density-model-name="_benchmark" 
```

Then we can run the inference steps as usual. However, we need to use the HessianNumerical option as analytical gradients of the known PDFs are not setup and scale the dataset to the number of effective events.

```
innfer --step="InitialFit" --benchmark="${benchmark}" --data-type="sim" --extra-infer-dir-name="ClosureTestFromBenchmark" --scale-to-eff-events --density-architecture="Benchmark" --extra-density-model-name="_benchmark" 
```

```
innfer --step="HessianNumerical" --benchmark="${benchmark}" --data-type="sim" --extra-infer-dir-name="ClosureTestFromBenchmark" --scale-to-eff-events --density-architecture="Benchmark" --extra-density-model-name="_benchmark" 
```

```
innfer --step="Covariance" --benchmark="${benchmark}" --data-type="sim" --extra-infer-dir-name="ClosureTestFromBenchmark" --scale-to-eff-events --density-architecture="Benchmark" --extra-density-model-name="_benchmark" 
```

### Inference with asimov

In order to get an expected (asimov) results, a similar workflow can be run for on the asimov dataset with --data-type="asimov".

```
innfer --step="InitialFit" --benchmark="${benchmark}" --data-type="asimov" --extra-infer-dir-name="Asimov"
```

```
innfer --step="HessianParallel" --benchmark="${benchmark}" --data-type="asimov" --extra-infer-dir-name="Asimov"
```

```
innfer --step="HessianCollect" --benchmark="${benchmark}" --data-type="asimov" --extra-infer-dir-name="Asimov"
```

The difference between the asimov and closure test instruction is here we do not need to correct the covariance with the D matrix. You can do this with the following command.

```
innfer --step="Covariance" --benchmark="${benchmark}" --data-type="asimov" --extra-infer-dir-name="Asimov"
```

### Summary plots

Now all the results have been run, we can display them.

```
innfer --step="Summary" --benchmark="${benchmark}" --extra-infer-dir-name="Asimov" --extra-infer-plot-name="with_closure_test" --summary-from="Covariance" --summary-nominal-name="Asimov" --summary-show-2sigma --summary-subtract --loop-over-nuisances --other-input="Closure Test:CovarianceWithDMatrixClosureTest:covariancewithdmatrix_results,Closure Test True Density:CovarianceClosureTestFromBenchmark:covariance_results"
```
