#!/bin/bash
cd /vols/cms/yhe4823/INNFER
ulimit -s unlimited
source env.sh
export PREP_DATA_DIR=/vols/cms/sbi_top_mass/data
export EVAL_DATA_DIR=data
export PLOTS_DIR=plots
export MODELS_DIR=models
export JOBS_DIR=None
python3 /vols/cms/yhe4823/INNFER/scripts/innfer.py --cfg="/vols/cms/yhe4823/INNFER/configs/run/btm_full_270526_sbi.py" --data-vs-simulation --extra-output-dir-name="DataVsSimStatOnly" --ratio-range="0.8,1.2" --sim-type="full" --specific-category="2223" --step="DistributionPlot"  --specific="file_name=combined;val_ind=1;category=2223"
