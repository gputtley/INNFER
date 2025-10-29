---
layout: page
title: "Step: PreProcess"
---

PreProcess is the largest of data preparation steps in **INNFER** repository. There are a number of techniques involved but overall the purpose of this step is to create datasets ready for training and testing density, regression and classifier models to use in building the likelihood and to produce validation datasets to validate the performance of the learned likelihood. It also creates a parameters yaml file which contains various information about the datasets including the names of the columns, the standardisation parameters, the information regarding the yields so we can produce a prediction for the total yield in the likelihood, and many other important features.

The methods used during the PreProcess step are:

- **Getting the yields**: 

It gets the nominal yield by calculating the sum of the weights at the default values of whichever base file is parsed as the "yield" for the particular "models" input in the configuration file. It also builds that dataset to get the 1 sigma variations of the nuisance parameters and save the sum of weights to find the relative yield effects of each nuisance, so this can be factored out and included as a log normal (lnN) nuisance parameter in the likelihood.

- **Get the binned fit inputs**: 

This method is only run if the "binned_fit_input" is defined in the configuration file. As **INNFER** is built for unbinned likelihood fits, this is only setup for a cross check of the unbinned results against the binned results. If you want to use a rigorous statistical framework for binned analysis, we recommend using [Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/). If you still wish to use binned fits in **INNFER**, this will calculate the yields and lnN effects in each bin and write this to the parameters file, for later use when building the binned likelihood.

- **Train, test and validation splitting**: 

This will split the base datasets, derived in the **LoadData** step into three datasets: a training and testing dataset for each model you wish to train and a validation dataset for every point you with to validate the likelihood estimation. The fraction of the events in the base datasets is parsed in the `preprocess:train_test_val_split` key of the configuration file and a colon separated string of 3 numbers that sum to 1, representing the fraction in `train`, `test` and `val` datasets, respectively.

There is also the option to drop specific values of a column from the `train` and `test` dataset but leave it in the `val` dataset. This is done with the `preprocess:drop_from_training` key in the configuration file, parsed as a dictionary with the column as key, linking to the values to drop in a list. In this case, all statistics are still used, as the unused events in the `train` and `test` dataset are moved to the `val` dataset. This can be useful for example, if you wanted to test the interpolation abilities of the learned likelihood from some discrete conditions.

The `train`, `test` and `val` datasets are also combined in a `full` dataset, in case you want to test the likelihoods ability against the `full` simulation. All dataset splits are done before any weight- or feature-based variations are performed.

- **Model variation**: 

This method performs the various weight and feature variations required for the required model `train` and `test` dataset and the `val` datasets. It also creates equivalent validation dataset (at the requested validation points) for the `train` and `test` splits called `train_inf` and `test_inf` (inf = inference), as well as for the `full` dataset. 
How the varied datasets are defined in the configuration is discussed [here](config.md).

The datasets are built slightly differently for the different purposes. The validation datasets are created by simply varying the weight and features to match that of the set of parameters defined. The training and testing datasets however, may require shifting to more than one value of a particular parameter. How these varied datasets are defined in the configuration is discussed [here](config.md). For density models, this is trivial as the relevant shifts define the whole dataset. For regression models, the effect on the weight are stored as a new column called `wt_shift`, which becomes the target for the regression. The classifiers, the shift dictates the variation of one class, the other class (reference) is created as the dataset and the default value.

- **Validation normalisation**:

The yields prediction for a particular key of the `models` is defined using the equations defined INSERT LINK TO EQUATIONS. As this can differ slightly from the sum of the weights of the datasets (if you are away from the interpolated points), we rescale the validation datasets to the prediction from the yield equations. This is the final manipulation of the validation datasets created. The final datasets are split up into three components: the observable (X), the parameter truth values (Y) and the weights (wt). The following parquet files are created in the data folder under `PreProcess/$MODEL_NAME/$CATEGORY/val_ind_$VAL_IND/`, where `$MODEL_NAME` is the key of the `models` input to the configuration, `$CATEGORY` is the category, and `$VAL_IND` is the index of the validation set of parameters. 

- **Get validation effective events**:

Here the number of effective events in each validation is determined as,

$$
N_{eff} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2},
$$

where $w_i$ is the weight per event $i$. This is calculated for each validation dataset and stored in the parameters file. It can later by used to calculate the statistical precision of a measurement using the statistics of the validation sample, rather than that of the Asimov.

- **Get validation binned histograms**:

This is another extension to perform binned likelihood fits. This defines the binned Asimov, at the various validation points, that can later be used for inference.

- **Flatten by yields**:

This method alters the weights of the `train` and `test` datasets. It does this to flatten the yields across conditional values in the training, so that no priority is given to a particular region in training. To do this, it divides the shifted weights by the yield equation, explained in INSERT LINK TO EQUATIONS.

- **Data Standardisation**:

In order to train neural networks (in particular normalising flows), it is crucial to standardise every column used in training to have a mean of 0 and a standard deviation of 1. Here the standardisation parameters are derived over a full pass of the `train` datasets for each model. Note, different models can have separate standardisation parameters. The `train` and `test` are then standardised, so that no data transformation is needed when training the models.

It is possible to parse a set of standardisation parameters to use in the configuration file. This can be useful if you want to re-preprocess your datasets, including an addition, but you do not want to retrain a particular model. Therefore, you can copy across the model trained with a particular set of standardisation parameters. This is done by adding a dictionary under `preprocess:standardisation`, with keys of the model name, the category, the model type, the column and the mean and the standardisations with the values you require. For information on this is discussed [here](config.md).

- **Classifier class balancing**:

If training classifier models to shift the nominal densities with the likelihood ratio trick, you will need to balance the classes of shifted variation vs reference value. This method normalises the weights of both classes to one another.

- **Shuffle training and testing datasets**:

The final manipulation of the `train` and `test` datasets is a shuffling. This can be a little slow as shuffling datasets that can only be loaded in batches, requires a few pass throughs of the data. Shuffling a dataset whilst only loading batch at a time can be very tricky, if you are noticing that your datasets are not sufficiently shuffled, you can increase the `--number-of-shuffles` parameter which by default is set to 10.

- **Make parameters file**:

This is the final method in this step. It saves a yaml file under `PreProcess/$MODEL_NAME/parameters.yaml` which contains important metadata about the datasets created. This includes a number of important information such as the 