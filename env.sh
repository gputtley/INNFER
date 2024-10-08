current_directory=$(pwd)
worker_pp="$current_directory/python/worker"
benchmarks_pp="$current_directory/python/worker/benchmarks"
runner_pp="$current_directory/python/runner"
bfp="$current_directory/python/BayesFlow"
export PYTHONPATH=${worker_pp}:${benchmarks_pp}:${runner_pp}:${bfp}:${PYTHONPATH}
if ! command -v nvidia-smi &> /dev/null ; then
  ulimit -s unlimited
  echo "Number of threads available $(nproc)"
  export TF_NUM_INTEROP_THREADS=$(nproc)
  export TF_NUM_INTRAOP_THREADS=$(nproc)
  export OMP_NUM_THREADS=$(nproc)
  export EVENTS_PER_BATCH=10000
  export EVENTS_PER_BATCH_FOR_GRADIENTS=4000
else
  export EVENTS_PER_BATCH=100000
  export EVENTS_PER_BATCH_FOR_GRADIENTS=40000
fi
export PLOTTING_CMS_LABEL="Work In Progress"
export PLOTTING_LUMINOSITY="\$138\ fb^{-1}\$"

ulimit -s unlimited
source miniconda/files/etc/profile.d/conda.sh
conda activate miniconda/files/envs/innfer_env
