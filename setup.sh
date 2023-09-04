current_directory=$(pwd)
pp="$current_directory/python"
export PYTHONPATH=${pp}:${PYTHONPATH}
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
