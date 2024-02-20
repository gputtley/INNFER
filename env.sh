current_directory=$(pwd)
pp="$current_directory/python"
bfp="$current_directory/BayesFlow"
export PYTHONPATH=${pp}:${bfp}:${PYTHONPATH}
#export PYTHONPATH=${pp}:${PYTHONPATH}
if ! command -v nvidia-smi &> /dev/null ; then
  echo "Number of threads available $(nproc)"
  export TF_NUM_INTEROP_THREADS=$(nproc)
  export TF_NUM_INTRAOP_THREADS=$(nproc)
  export OMP_NUM_THREADS=$(nproc)  

  # May need to be set on some machines
  #export TF_NUM_INTRAOP_THREADS=1
  #export TF_ENABLE_ONEDNN_OPTS=0
fi
ulimit -s unlimited
source miniconda/files/etc/profile.d/conda.sh
conda activate miniconda/files/envs/innfer_env
