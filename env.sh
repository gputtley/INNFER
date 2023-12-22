current_directory=$(pwd)
pp="$current_directory/python"
export PYTHONPATH=${pp}:${PYTHONPATH}
if ! command -v nvidia-smi &> /dev/null ; then
  export TF_NUM_INTEROP_THREADS=1
  export TF_NUM_INTRAOP_THREADS=1
  export TF_ENABLE_ONEDNN_OPTS=0
fi
ulimit -s unlimited
conda activate innfer_env
