# Runtime setup independent of the host checkout and conda activation.
export INNFER_ROOT=/opt/innfer/app
export INNFER_ENV=/opt/innfer/env
export PATH="/opt/innfer/bin:$INNFER_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$INNFER_ROOT/python/worker:$INNFER_ROOT/python/worker/benchmarks:$INNFER_ROOT/python/runner:$INNFER_ROOT/python/BayesFlow:$INNFER_ROOT/python/runner/custom_module:$INNFER_ROOT/python/worker/custom"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/innfer-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/innfer-cache}"
export PREP_DATA_DIR="${PREP_DATA_DIR:-data}"
export EVAL_DATA_DIR="${EVAL_DATA_DIR:-data}"
export MODELS_DIR="${MODELS_DIR:-models}"
export PLOTS_DIR="${PLOTS_DIR:-plots}"
export JOBS_DIR="${JOBS_DIR:-jobs}"
export PLOTTING_CMS_LABEL="${PLOTTING_CMS_LABEL:-Work In Progress}"
export PLOTTING_LUMINOSITY="${PLOTTING_LUMINOSITY:-\$138\ fb^{-1} (13\ TeV)\$}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

if command -v nvidia-smi >/dev/null 2>&1; then
  export EVENTS_PER_BATCH_FOR_LOADDATA="${EVENTS_PER_BATCH_FOR_LOADDATA:-100000}"
  export EVENTS_PER_BATCH_FOR_PREPROCESS="${EVENTS_PER_BATCH_FOR_PREPROCESS:-1000000}"
  export EVENTS_PER_BATCH_FOR_SHUFFLE="${EVENTS_PER_BATCH_FOR_SHUFFLE:-10000000}"
  export EVENTS_PER_BATCH="${EVENTS_PER_BATCH:-100000}"
  export EVENTS_PER_BATCH_FOR_GRADIENTS="${EVENTS_PER_BATCH_FOR_GRADIENTS:-10000}"
  export EVENTS_PER_BATCH_FOR_HESSIAN="${EVENTS_PER_BATCH_FOR_HESSIAN:-5000}"
else
  export EVENTS_PER_BATCH_FOR_LOADDATA="${EVENTS_PER_BATCH_FOR_LOADDATA:-50000}"
  export EVENTS_PER_BATCH_FOR_PREPROCESS="${EVENTS_PER_BATCH_FOR_PREPROCESS:-200000}"
  export EVENTS_PER_BATCH_FOR_SHUFFLE="${EVENTS_PER_BATCH_FOR_SHUFFLE:-2000000}"
  export EVENTS_PER_BATCH="${EVENTS_PER_BATCH:-10000}"
  export EVENTS_PER_BATCH_FOR_GRADIENTS="${EVENTS_PER_BATCH_FOR_GRADIENTS:-4000}"
  export EVENTS_PER_BATCH_FOR_HESSIAN="${EVENTS_PER_BATCH_FOR_HESSIAN:-2000}"
  export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-$(nproc)}"
  export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$(nproc)}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
fi

# Explicit CUDA data location avoids XLA depending on host conda paths.
case "${XLA_FLAGS:-}" in
  *--xla_gpu_cuda_data_dir*) ;;
  *)
    for innfer_cuda_dir in "$INNFER_ENV"/lib/python*/site-packages/nvidia/cuda_nvcc; do
      if [ -f "$innfer_cuda_dir/nvvm/libdevice/libdevice.10.bc" ]; then
        export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$innfer_cuda_dir"
        break
      fi
    done
    ;;
esac
unset innfer_cuda_dir
