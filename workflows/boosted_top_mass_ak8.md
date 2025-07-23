# Boosted Top Mass

## Find the files to go in the configuration files

```
python3 scripts/config_inputs.py -f "data/top_reco/230725/TT*.parquet" -ac "mass,166p5,166.5" -ac "mass,169p5,169.5" -ac "mass,171p5,171.5" -ac "mass,173p5,173.5" -ac "mass,175p5,175.5" -ac "mass,178p5,178.5" -ac "run,201,2" -ac "run,202,3" -dc "mass,172.5"
```

```
python3 scripts/config_inputs.py -f "data/top_reco/230725/W*.parquet,data/top_reco/170725/ST*.parquet,data/top_reco/170725/DY*.parquet" -ac "run,201,2" -ac "run,202,3"
```

## Set configuration file

Now we have made the config, lets set the configuration file.

```
cfg="btm_180725.yaml"
```

## SnakeMake running

The full workflow can be run with snakemake. This will utilise the inputs and outputs of each job to submit each job only when the inputs are available. To do this, open up a new terminal (recommend tmux or screen) and run this following command with the relevant snakemake submission configuration. NOT SETUP YET FOR THIS EXAMPLE.

```
python3 scripts/innfer.py --step="SnakeMake" --cfg="${cfg}" --snakemake-cfg="btm_5Y_condor_ic.yaml"
```

## Step-by-step running

These instructions are the step-by-step commands to be ran if you are not using the snakemake workflow. These can be particularly useful for debugging. Note each step can be submitted to a batch cluster by using the --submit="condor.yaml" for example. This will split the jobs as finely as possible. If you want to group these together you can use the --points-per-job option.

### PreProcess datasets

A few steps are needed preprocess the relevant training, testing and validation. Firstly, the base files are created.

```
python3 scripts/innfer.py --step="LoadData" --cfg="${cfg}"
```

The training, testing and validation files, as well as the parameter files are created with this PreProcess step.

```
python3 scripts/innfer.py --step="PreProcess" --cfg="${cfg}"
```

Finally, in order to optimally fraction the breit-wigner reweighting (specific to tops), you can run the custom module for this.

```
python3 scripts/innfer.py --step="Custom" --cfg="${cfg}" --custom-module="top_bw_fractioning" --custom-options="gen_mass=GenTop1_mass;gen_mass_other=GenTop2_mass"
```

### Visualising datasets

To visualise these datasets there are two modules, one for training and testing datasets, and one for validation datasets.

```
python3 scripts/innfer.py --step="InputPlotTraining" --cfg="${cfg}"
```
```
python3 scripts/innfer.py --step="InputPlotValidation" --cfg="${cfg}"
```

### Training networks

The next step is to train the density network. Here we use a BayesFlow density model. If you are going to submit this to a batch, it is recommended you add the --disable-tqdm option.

```
python3 scripts/innfer.py --step="TrainDensity" --cfg="${cfg}" --density-architecture="density_default.yaml"
```

In this case, we are using regression models for weight variations. This training can be performed with a similar command.

```
python3 scripts/innfer.py --step="TrainRegression" --cfg="${cfg}" --density-architecture="regression_default.yaml"
```

### Density performance metrics

To test the performance of the density model, there is a module to get some performance metrics of the learned density. This is also utilised in the HyperparameterScan and BayesianHyperparameterTuning steps. To choose what performance metrics to get you can parse the --density-performance-metrics option, providing a comma separated list of any combination of loss, histogram, multidim, and inference.

```
python3 scripts/innfer.py --step="DensityPerformanceMetrics" --cfg="${cfg}"
```

### Regression evaluation and plotting

Although there are currently no performance metrics implemented for the regression models, it is useful to look at distributions of the average weights (y dataset for regression) and the average regressed values, as a function of each input. To do this we first evaluate the regression model on the training and testing datasets. This step provides an additional benefit of renormalising the regression models. This is done by fitting a spline and is used later in the asimov created and density estimation.

```
python3 scripts/innfer.py --step="EvaluateRegression" --cfg="${cfg}"
```

Once you have created the regression evaluation parquet, the plots can be made with the following code.

```
python3 scripts/innfer.py --step="PlotRegression" --cfg="${cfg}"
```

### Make asimov datasets

To create an asimov dataset parquet file the following command is available. This is utilised in the generator code and the inference code when using the option --data-type="asimov". There are two options to choose the number of events. The first is to parse --number-of-asimov-events=1000000 (default), this will create this many asimov events for each model and validation point. The second is --use-asimov-scaling=10, this will create 10 times the asimov events the the expected yield for that model.

```
python3 scripts/innfer.py --step="MakeAsimov" --cfg="${cfg}" --use-asimov-scaling=10
```

### Generator plots

To test the distribution of the asimov events versus the validation datasets, you can plot the distribution of each with respect to the input variables with the following command. You can also draw a 2d unrolled plot with the --plot-2d-unrolled, or the standardised versions of the variables with --plot-transformed.

```
python3 scripts/innfer.py --step="Generator" --cfg="${cfg}"
```

To see a summary of all validation points on the same plot, you can use the following command. There is also the --val-inds option to choose the specific validation indices you want on the plot.

```
python3 scripts/innfer.py --step="GeneratorSummary" --cfg="${cfg}"
```

### p-value plots

These steps are to test whether you can separate the learned and sampled synthetic (asimov) dataset from the simulated validation datasets. This is done in the context of a p-value. Firstly, performance metrics are found to attempt to separate the simulated and synthetic dataset. 

```
python3 scripts/innfer.py --step="PValueSimVsSynth" --cfg="${cfg}"
```

However, these metrics have no understanding on their own. Therefore, we build up a dataset of these metrics attempting to separate two randomly sampled synthetic dataset (with the same stats as the simulated sample). The number of toys thrown can be parsed to this function. 

```
python3 scripts/innfer.py --step="PValueSynthVsSynth" --cfg="${cfg}" --number-of-toys=100
```

These results will need to be collected into one file.

```
python3 scripts/innfer.py --step="PValueSynthVsSynthCollect" --cfg="${cfg}" --number-of-toys=100
```

Finally, we can plot where the simulated verses synthetic metrics fall in the synthetic versus synthetic distribution and calculate a p-value.

```
python3 scripts/innfer.py --step="PValueDatasetComparisonPlot" --cfg="${cfg}"
```

### Inference closure test

Now, we test the performance of the learned density on the validation datasets. This requires a number of steps. First of which is an initial minimisation of the likelihood. There are a number of options to consider here. The --likelihood-type can be chosen to be unbinned or unbinned_extended depending on whether you want to include the total event count as a Poisson term. There are also binned fit options, however, this requires additional configuration inputs during preprocessing which will not be documented here. The --minimisation-method can also be chosen, typically best to perform this with minuit. The --freeze option will freeze the parameter of key to value provided. There is also an option to not use the regression models. These are --only-density to skip the regression models and --skip-non-density to skip the validation indices that would require the regression models. There are other options beyond this that can also be parsed.

```
python3 scripts/innfer.py --step="InitialFit" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

The second step is to calculate the Hessian matrix. There are a few ways to do this in the code, the way shown below is to parallelise the calculation (useful for batch submission) as much as possible and collect. This uses the gradients straight from the models. There is also the --step="Hessian" option to do this without parallelising. The other choice is to calculate this numerically and can be done with the --step="HessianNumerical" option.

```
python3 scripts/innfer.py --step="HessianParallel" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

```
python3 scripts/innfer.py --step="HessianCollect" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

The next steps are to correct for the true statistical uncertainty of the validation dataset, rather than the statistical uncertainty of the predicted yield. We calculate the D Matrix using the gradients directly from the model. 

```
python3 scripts/innfer.py --step="DMatrix" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

The D Matrix corrected covariance matrix and the intervals are then calculated with the following command.

```
python3 scripts/innfer.py --step="CovarianceWithDMatrix" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTest"
```

### Inference closure test with others frozen

The previous inference closure test instructions involve floating all available parameters simultaneous. This set of instructions details the closure test when performing stat. only fits for each parameter, i.e. freezing all but the one parameter. By default all shape POIs will be looped over, but if you want the shape nuisances (--loop-over-nuisances), rate parameters (--loop-over-rates) and lnN parameters (--loop-over-lnN) these options are available.  

```
python3 scripts/innfer.py --step="InitialFit" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTestFrozen" --freeze="all-but-one" --loop-over-nuisances --loop-over-rates --loop-over-lnN
```

```
python3 scripts/innfer.py --step="HessianParallel" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTestFrozen" --freeze="all-but-one" --loop-over-nuisances --loop-over-rates --loop-over-lnN
```

```
python3 scripts/innfer.py --step="HessianCollect" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTestFrozen" --freeze="all-but-one" --loop-over-nuisances --loop-over-rates --loop-over-lnN
```

```
python3 scripts/innfer.py --step="DMatrix" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTestFrozen" --freeze="all-but-one" --loop-over-nuisances --loop-over-rates --loop-over-lnN
```

```
python3 scripts/innfer.py --step="CovarianceWithDMatrix" --cfg="${cfg}" --data-type="sim" --extra-infer-dir-name="ClosureTestFrozen" --freeze="all-but-one" --loop-over-nuisances --loop-over-rates --loop-over-lnN
```

Finally, it is useful (for plotting purposes) to collect the results into one directory. The following code will do this.

```
python3 scripts/innfer.py --step="SummaryAllButOneCollect" --cfg="${cfg}" --extra-infer-dir-name="ClosureTestFrozen" --summary-from="CovarianceWithDMatrix" --loop-over-nuisances --loop-over-rates --loop-over-lnN
```

### Inference with asimov

In order to get an expected (asimov) results, a similar workflow can be run for on the asimov dataset with --data-type="asimov". This can be very time consuming and so this is not typically ran for all validation points. Therefore, we specify --specific="file_name=combined;val_ind=0" here.

```
python3 scripts/innfer.py --step="InitialFit" --cfg="${cfg}" --data-type="asimov" --extra-infer-dir-name="Asimov" --specific="file_name=combined;val_ind=0"
```

```
python3 scripts/innfer.py --step="HessianParallel" --cfg="${cfg}" --data-type="asimov" --extra-infer-dir-name="Asimov" --specific="file_name=combined;val_ind=0"
```

```
python3 scripts/innfer.py --step="HessianCollect" --cfg="${cfg}" --data-type="asimov" --extra-infer-dir-name="Asimov" --specific="file_name=combined;val_ind=0"
```

The difference between the asimov and closure test instruction is here we do not need to correct the covariance with the D matrix. You can do this with the following command.

```
python3 scripts/innfer.py --step="Covariance" --cfg="${cfg}" --data-type="asimov" --extra-infer-dir-name="Asimov" --specific="file_name=combined;val_ind=0"
```

### Summary plots

Now all the results have been run, we can display them. There a few ways to do this. First, is a summary of both closure test for all validation indices. Note that this plot will become untenable for a large number of validation points or parameters.

```
python3 scripts/innfer.py --step="Summary" --cfg="${cfg}" --extra-infer-dir-name="ClosureTest" --extra-infer-plot-name="with_frozen" --summary-from="CovarianceWithDMatrix" --summary-nominal-name="Closure Test" --summary-show-2sigma --summary-subtract --loop-over-nuisances --other-input="Closure Test (Others Frozen):CovarianceWithDMatrixClosureTestFrozen:covariancewithdmatrix_results"
```

Next are the summary plots per validation point. This creates a plot of the parameters (rates and lnN optional) for each validation point. This is firstly done individually for the closure test, the closure test (others frozen), and the asimov.

```
python3 scripts/innfer.py --step="SummaryPerVal" --cfg="${cfg}" --extra-infer-dir-name="ClosureTest" --summary-from="CovarianceWithDMatrix" --summary-nominal-name="Closure Test" --summary-show-2sigma --loop-over-rates --loop-over-lnN
```

```
python3 scripts/innfer.py --step="SummaryPerVal" --cfg="${cfg}" --extra-infer-dir-name="ClosureTestFrozen" --summary-from="CovarianceWithDMatrix" --summary-nominal-name="Closure Test (Others Frozen)" --summary-show-2sigma --loop-over-rates --loop-over-lnN
```

```
python3 scripts/innfer.py --step="SummaryPerVal" --cfg="${cfg}" --extra-infer-dir-name="Asimov" --summary-from="Covariance" --summary-nominal-name="Asimov" --summary-show-2sigma --loop-over-rates --loop-over-lnN --specific="file_name=combined;val_ind=0"
```

This can also be done as a summary of the closure test and the closure test (others frozen).

```
python3 scripts/innfer.py --step="SummaryPerVal" --cfg="${cfg}" --extra-infer-dir-name="ClosureTest" --extra-infer-plot-name="with_frozen" --summary-from="CovarianceWithDMatrix" --summary-nominal-name="Closure Test" --summary-show-2sigma --loop-over-rates --loop-over-lnN --other-input="Closure Test (Others Frozen):CovarianceWithDMatrixClosureTestFrozen:covariancewithdmatrix_results"
```

And also done for closure test, the closure test (others frozen), and the asimov.

```
python3 scripts/innfer.py --step="SummaryPerVal" --cfg="${cfg}" --extra-infer-dir-name="Asimov" --extra-infer-plot-name="with_closure" --summary-from="Covariance" --summary-nominal-name="Asimov" --summary-show-2sigma --loop-over-rates --loop-over-lnN --other-input="Closure Test:CovarianceWithDMatrixClosureTest:covariancewithdmatrix_results,Closure Test (Others Frozen):CovarianceWithDMatrixClosureTestFrozen:covariancewithdmatrix_results" --specific="file_name=combined;val_ind=0"
```