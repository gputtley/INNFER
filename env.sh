current_directory=$(pwd)
worker_pp="$current_directory/python/worker"
runner_pp="$current_directory/python/runner"
bfp="$current_directory/BayesFlow"
export PYTHONPATH=${worker_pp}:${runner_pp}:${bfp}:${PYTHONPATH}
if ! command -v nvidia-smi &> /dev/null ; then
  echo "Number of threads available $(nproc)"
  export TF_NUM_INTEROP_THREADS=$(nproc)
  export TF_NUM_INTRAOP_THREADS=$(nproc)
  export OMP_NUM_THREADS=$(nproc)  
fi
ulimit -s unlimited
source miniconda/files/etc/profile.d/conda.sh
conda activate miniconda/files/envs/innfer_env
