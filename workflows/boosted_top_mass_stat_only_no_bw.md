# Boosted Top Mass Stat Only No BW

## Set configuration file

Firstly, lets set the configuration file.

```
cfg="btm_070125_2d_stat_only_no_bw.yaml"
```

## PreProcess datasets

Next we preprocess, remove outliets, perform bw reweighting and split up the validation files. We run this locally on an interactive gpu. These cannot be run with snakemake

```
python3 scripts/innfer.py --step=PreProcess,RemoveOutliers,SplitValidationFiles --cfg="${cfg}"
```

## Train datasets

Next we train the models of the IC GPU nodes. We can do this on snakemake, however, I find it is better to check the performance metrics of the model before submitting the full validation methods. We also change the architectures slightly for the two models.


```
python3 scripts/innfer.py --step=Train --cfg="${cfg}" --disable-tqdm --use-wandb --wandb-project-name=BTM_stat_only --specific=ttbar --overwrite-architecture="epochs=20" --submit=condor_gpu.yaml
```
```
python3 scripts/innfer.py --step=Train --cfg="${cfg}" --disable-tqdm --use-wandb --wandb-project-name=BTM_stat_only --specific=other --overwrite-architecture="epochs=20,batch_size=512" --submit=condor_gpu.yaml
```

## Performance metrics

As I mentioned, I next check the performance metrics.

```
python3 scripts/innfer.py --step=PerformanceMetrics --cfg="${cfg}" --submit=condor_gpu.yaml
```

## Validation snakemake

Finally, I run the full validation snakemake workflow. I typically do this in a tmux session. Remember to reset the configuration file if starting a new terminal.

```
python3 scripts/innfer.py --step=SnakeMake --cfg="${cfg}" --snakemake-cfg="validation_btm_stat_only"
```