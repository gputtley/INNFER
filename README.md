# INNFER - Invertible Neural Networks for Extracting Results

## Installing Repository and Packages

To import the github repository, clone with the following command.

```
git clone https://github.com/gputtley/INNFER.git
```

The bayesflow package is needed for this repository. This can be installed with the below command. Be careful when running on a remote shared machine. Where possible use a conda environment and export your tmp directory to a location with enough storage before installing. 

```
pip3 install bayesflow
```

## Setup Environment

At the beginning of every session you will need to run the following command to setup the environment.

```
source env.sh
```

## Run the Basic Example

The simple example of inferring the top mass from a mass resolution like variable can be run with this command.

```
python3 scripts/top_mass_basic_example.py
```

There are a number of running options within the script, such as to add a background and vary the signal yield.

For longer jobs you may want to parallelise parts of the job a batch, such as an SGE cluster. For this you can run the following commands.

```
python3 scripts/top_mass_basic_example.py --submit="SGE" --use-signal-fraction --skip-closure --skip-probability --skip-generation --skip-inference
```
```
for mass in 171.0 172.0 173.0 174.0; do
for frac in 0.1 0.2 0.3; do
python3 scripts/top_mass_basic_example.py --submit="SGE" --use-signal-fraction --skip-initial-distribution --load-model --plot-true-masses=${mass} --plot-signal-fractions=${frac}
done
done
```