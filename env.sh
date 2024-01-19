current_directory=$(pwd)
pp="$current_directory/python"
export PYTHONPATH=${pp}:${PYTHONPATH}
if ! command -v nvidia-smi &> /dev/null ; then
  num_threads=$(nproc)
  export TF_NUM_INTEROP_THREADS=$num_threads
  # May not to be set 
  #export TF_NUM_INTRAOP_THREADS=1
  #export TF_ENABLE_ONEDNN_OPTS=0
fi
ulimit -s unlimited
source miniconda/files/etc/profile.d/conda.sh
conda activate miniconda/files/envs/innfer_env