# INNFER - Invertible Neural Networks for Extracting Results

## Installing Repository and Packages

To import the github repository, clone with the following command.

```
git clone https://github.com/gputtley/INNFER.git
```

Then to set up the conda environment for running this repository run this command. You will need to click `enter` and `yes` through all prompts. For the conda installation license hit `q` to go the end of the file and then you can fit `enter`. 

```
source setup.sh
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
For longer jobs you may want to parallelise parts of the job a batch, such as on an SGE cluster. 

For instance, to run with the `--add-background` option, you can use the following commands.
```
python3 scripts/top_mass_basic_example.py --submit="SGE" --add-background --skip-closure --skip-probability --skip-generation --skip-inference
```
```
for mass in 171.0 172.0 173.0 174.0; do
  python3 scripts/top_mass_basic_example.py --submit="SGE" --add-background --skip-initial-distribution --load-model --plot-true-masses=${mass}
done
```

To run with the `--use-signal-fraction` on a cluster, you can use the following commands.

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