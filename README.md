# Installing Repository and Packages

To import the github repository, clone with the following command.

```
git clone https://github.com/gputtley/BayesFlowForInference.git
```

The bayesflow package is needed for this repository. This can be installed with the below command. Be careful when running on a remote shared machine. Where possible use a conda environment and export your tmp directory to a location with enough storage before installing. 

```
pip3 install bayesflow
```

# Setup Environment

At the beginning of every session you will need to run the following command to setup the environment.

```
source setup.sh
```

# Run the Basic Example

The simple example of inferring the top mass from a mass resolution like variable can be run with this command.

```
python3 scripts/top_mass_basic_example.py
```