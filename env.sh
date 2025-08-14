current_directory=$(pwd)
worker_pp="$current_directory/python/worker"
benchmarks_pp="$current_directory/python/worker/benchmarks"
custom_pp="$current_directory/python/runner/custom_module"
runner_pp="$current_directory/python/runner"
bfp="$current_directory/python/BayesFlow"
export PYTHONPATH=${worker_pp}:${benchmarks_pp}:${runner_pp}:${bfp}:${custom_pp}:${PYTHONPATH}
if ! command -v nvidia-smi &> /dev/null ; then
  ulimit -s unlimited
  echo "Number of threads available $(nproc)"
  export TF_NUM_INTEROP_THREADS=$(nproc)
  export TF_NUM_INTRAOP_THREADS=$(nproc)
  export OMP_NUM_THREADS=$(nproc)
  export EVENTS_PER_BATCH_FOR_PREPROCESS=1000000
  export EVENTS_PER_BATCH=10000
  export EVENTS_PER_BATCH_FOR_GRADIENTS=4000
  export EVENTS_PER_BATCH_FOR_HESSIAN=2000
  ulimit -s unlimited
else
  export EVENTS_PER_BATCH_FOR_PREPROCESS=10000000
  export EVENTS_PER_BATCH=50000
  export EVENTS_PER_BATCH_FOR_GRADIENTS=10000
  export EVENTS_PER_BATCH_FOR_HESSIAN=5000
fi
export PLOTTING_CMS_LABEL="Work In Progress"
export PLOTTING_LUMINOSITY="\$138\ fb^{-1} (13\ TeV)\$"

export DATA_DIR="data"
export MODELS_DIR="models"
export PLOTS_DIR="plots"

ulimit -s unlimited
source miniconda/files/etc/profile.d/conda.sh
conda activate miniconda/files/envs/innfer_env
alias innfer="$PWD/scripts/innfer.py"