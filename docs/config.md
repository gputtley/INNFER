---
layout: page
title: "Configuration File"
---

# Building the Configuration

This section shows how to define and customise your input configuration file. The configuration file may be parsed in two formats: one is a `.yaml` file, and the other is a `.py` file where the dictionary being parsed is stored as the `config` variable name.

## Input Configuration File

These instruction describes the structure and purpose of each key within the `config` dictionary. The configuration defines how to build, train and validate density, classifier and regression models, as well as how to combine them to build the likelihood functions. Examples of this is shown in the `configs/run` directory.

## Top-Level Keys

### `name`
- **Type:** `str`
- **Description:** A unique identifier for the configuration or training run. This name will be used as the directory name that your data, models and plots are written to.
- **Example:** `"BTM_241025`

### `variables`
- **Type:** `list[str]`
- **Description:** The list of observables/input features used in training and likelihood model evaluation.
- **Example:** ```["CombinedSubJets_mass", "CombinedSubJets_pt"]```

### `pois`
- **Type:** `list[str]`
- **Description:** Parameters of interest (POIs) for the fit, typically physical quantities such as the top-quark mass.
- **Example:** ```["bw_mass"]```

### `nuisances`
- **Type:** `list[str]`
- **Description:** Names of all nuisance parameters included in the model (e.g., systematic uncertainties).
- **Example:** ```["jec"]```

### `categories`
- **Type:** `dict[str, str]`
- **Description:** A mapping of analysis categories to selection expressions strings. If there is no need for a selection, you can parse `None`
- **Example:** ```{"run2": "(run==2.0)", "run3": "(run==3.0)"}```

### `data_file`
- **Type:** `str` or `None`
- **Description:** Path to the main data file, if applicable. `None` if data are not directly included in this configuration. Data can be parsed as a root or a parquet file.

### `inference`
- **Type:** `dict`
- **Description:** Configuration of the inference or statistical fitting stage.
- **Keys:**
  - **nuisance_constraints:** List of nuisances to add constraints to in the fit. This constaint will be a unit Gaussian.
  - **rate_parameters:** Names of rate normalisation parameters.
  - **lnN:** Dictionary of processes and lists of log-normal uncertainties to add to the fit.
  - **binned_fit:** Dictionary for binned fit configuration, if needed.

### `default_values`
- **Type:** `dict[str,float]`
- **Desciption:** Default central values for parameters of interest. Can also set default value of nuisances and rate parameters, but if not given, this is set to 0 for nuisances and 1 for rate parameters.
- **Example:** ```{"bw_mass": 172.5}```

### `models`
- **Type:** `dict`
- **Description:** Definitions of all model components by process type. This is done per process.
- **Structure:**
```python
{
  "PROCESS_NAME": {
    "density_models": [...],
    "classifier_models": [...],
    "regression_models": [...],
    "yields": [...]
  }
}
```
Each sublist (e.g., density_models) contains entries with dictionary defining the models with the following keys:

| Key          | Type      | Description                                                  |
| ------------ | --------- | ------------------------------------------------------------ |
| `parameters` | list[str] | Model parameters being learned or varied (e.g. `"bw_mass"`). |
| `file`       | str       | Reference to a base file defined in `files`.                 |
| `shifts`     | dict      | Specifies parameter variations (type, range, and options).   |
| `n_copies`   | int       | Number of training copies of the dataset taken.              |
| `categories` | list[str] | List of categories this model applies to.                    |

- **Example of the sublist entry:**
```python
{
  "parameters": ["bw_mass"],
  "file": "base_ttbar_run2",
  "shifts": {
    "bw_mass": {"type": "flat_top", "range": [169.5, 175.5], "other": {"sigma_out": 1.0}}
  },
  "n_copies": 5,
  "categories": ["run2"]
}
```

### `validation`
- **Type:** `dict`
- **Description:** Defines validation configurations and datasets for testing model predictions.
- **Keys:** 
  - **loop:** List of parameter combinations to validate on.
  - **files:** Mapping of process types to the files and categories used for validation.
- **Example:**
```python
"validation": {
  "loop": [{"bw_mass": 171.5}, {"bw_mass": 172.5}, {"bw_mass": 173.5}],
  "files": {
    "ttbar": [{"file": "base_ttbar_run2", "categories": ["run2"]}],
    "other": [{"file": "base_other_run2", "categories": ["run2"]}]
  }
}
```

### `preprocess`
- **Type:** `dict`
- **Description:** Configuration for preprocessing and data preparation.
- **Main keys:**
  - **train_test_val_split:** Ratio for train/test/validation datasets (e.g. "0.8:0.1:0.1").
  - **save_extra_columns:** Extra variables to save for different process types. This can be useful for custom modules.
  - **standardisation:** Precomputed mean and standard deviation values for standardising features per category and process.
- **Example:**
```python
"preprocess": {
  "train_test_val_split": "0.8:0.1:0.1",
  "save_extra_columns": {"ttbar": ["sim_mass","GenTop1_mass"], "other": []},
  "standardisation": {
    "ttbar": {
      "run2": {
        "density": {
          "CombinedSubJets_mass": {"mean": 140.0, "std": 45.3},
          "CombinedSubJets_pt": {"mean": 490.0, "std": 94.6}
        }
      }
    }
  }
}
```

### `files`
- **Type:** `dict[str, dict]`
- **Description:** Defines all base input datasets and their configurations for each category and process type. These are referenced by name in the models section. Each file entry includes:
| Key                        | Type            | Description                                                        |
| -------------------------- | --------------- | ------------------------------------------------------------------ |
| `inputs`                   | list[str]       | Paths to Parquet files containing event data. Inputs can be root or a parquet file. |
| `add_columns`              | dict[str, list] | Extra metadata columns such as `sim_mass`, `run`, and `year_ind`.  |
| `selection`                | str             | Pre-selection cut applied before training.                         |
| `weight`                   | str             | Name of the event weight column.                                   |
| `parameters`               | list            | Parameters defined for the dataset (usually empty here).           |
| `pre_calculate`            | dict            | Expressions for derived quantities calculated before training.     |
| `post_calculate_selection` | str             | Additional selection applied after derived variables are computed. |
| `weight_shifts`            | dict            | Expressions defining weight variations for systematic studies.     |
- **Example:**
```python
"base_ttbar_run2": {
  "inputs": ["TTToSemiLeptonic_2018.parquet", "TTToSemiLeptonic_2022_preEE.parquet"],
  "add_columns": {"sim_mass": [172.5, 172.5], "run": [2.0, 3.0]},
  "selection": "((JetLepton_deltaR>0.25) & (JetLepton_ptrel>30))",
  "weight": "weight",
  "parameters": [],
  "pre_calculate": {},
  "post_calculate_selection": "(CombinedSubJets_pt > 400)",
  "weight_shifts": {"bw_mass": "(...)"},
}
```

<br>

---

Next: [SnakeMake](snakemake.md).