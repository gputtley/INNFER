#!/bin/bash
cd /vols/cms/yhe4823/INNFER
ulimit -s unlimited
source env.sh
export PREP_DATA_DIR=/vols/cms/sbi_top_mass/data
export EVAL_DATA_DIR=data
export PLOTS_DIR=plots
export MODELS_DIR=models
export JOBS_DIR=None
python3 /vols/cms/yhe4823/INNFER/scripts/innfer.py --cfg="/vols/cms/yhe4823/INNFER/configs/run/btm_full_270526_sbi.py" --disable-tqdm --specific-category="2223" --step="PreProcessParallelNuisanceVariations"  --specific="file_name=ttbar;category=2223;nuisance=electron_trigger;shift=up"
python3 /vols/cms/yhe4823/INNFER/scripts/innfer.py --cfg="/vols/cms/yhe4823/INNFER/configs/run/btm_full_270526_sbi.py" --disable-tqdm --specific-category="2223" --step="PreProcessParallelNuisanceVariations"  --specific="file_name=ttbar;category=2223;nuisance=electron_trigger;shift=down"
