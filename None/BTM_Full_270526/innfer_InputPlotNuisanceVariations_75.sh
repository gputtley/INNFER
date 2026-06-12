#!/bin/bash
cd /vols/cms/yhe4823/INNFER
ulimit -s unlimited
source env.sh
export PREP_DATA_DIR=/vols/cms/sbi_top_mass/data
export EVAL_DATA_DIR=data
export PLOTS_DIR=plots
export MODELS_DIR=models
export JOBS_DIR=None
python3 /vols/cms/yhe4823/INNFER/scripts/innfer.py --cfg="/vols/cms/yhe4823/INNFER/configs/run/btm_full_270526_sbi.py" --sim-type="full" --specific-category="2223" --step="InputPlotNuisanceVariations"  --specific="file_name=other;category=2223;nuisance=RelativeBal"
python3 /vols/cms/yhe4823/INNFER/scripts/innfer.py --cfg="/vols/cms/yhe4823/INNFER/configs/run/btm_full_270526_sbi.py" --sim-type="full" --specific-category="2223" --step="InputPlotNuisanceVariations"  --specific="file_name=other;category=2223;nuisance=RelativeStatHF_2022_postEE"
