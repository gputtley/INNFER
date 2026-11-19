mkdir -p tmp
export TMPDIR="./tmp/"
export TEMP=$TMPDIR
export TMP=$TMPDIR
export CONDA_PKGS_DIRS=$TMPDIR
export PIP_CACHE_DIR=$TMPDIR
export XDG_CACHE_HOME=$TMPDIR
MINICONDA_BASE="${MINICONDA_BASE:-./miniconda}"


if [ $# -eq 0 ] || [ "$1" == "conda" ]; then
  echo "Installing miniconda"
  MINICONDA_BASE="${MINICONDA_BASE:-./miniconda}"
  MINICONDA_BASE="$(realpath -m "$MINICONDA_BASE")"
  CONDA_DIR="$MINICONDA_BASE/files"
  mkdir -p "$MINICONDA_BASE"
  pushd "$MINICONDA_BASE"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh 
  ./Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda update -y conda
  rm Miniconda3-latest-Linux-x86_64.sh
  popd
fi

if [ $# -eq 0 ] || [ "$1" == "env" ]; then
  echo "Creating enviroment"
  source "${MINICONDA_BASE}/files/etc/profile.d/conda.sh"
  conda clean -a -y
  conda config --set channel_priority flexible
  conda env create --file=configs/setup/environment.yaml
  conda activate innfer_env
  # Need to do this because of wrapt version conflict between tensorflow and snakemake
  pip3 install tensorflow[and-cuda]==2.15.1 wrapt==1.14.1
  pip3 install --no-deps snakemake==9.9.0
  pip3 install --no-deps snakemake-interface-common==1.21.0 snakemake-interface-executor-plugins==9.3.9 snakemake-interface-report-plugins==1.2.0 yte==1.9.0 snakemake-interface-storage-plugins==4.2.3 snakemake-interface-logger-plugins==1.2.4
  chmod +x scripts/innfer.py
  alias innfer="$PWD/scripts/innfer.py"
  cd python/BayesFlow
  pip install -e .
  cd ../../
fi

if [ $# -eq 0 ] || [ "$1" == "snakemake_ic" ] || [ "$1" == "snakemake_cern" ] || [ "$1" == "snakemake_condor" ]; then
  _SetupSnakeMake() {
    local setup_snakemake_target="$1"
    local setup_profile="profiles/ic"
    local setup_command

    source "${MINICONDA_BASE}/files/etc/profile.d/conda.sh" || return 1
    conda activate innfer_env || return 1

    # The environment already includes this plugin. Install only if missing.
    if ! python3 -c 'from importlib.metadata import distribution; distribution("snakemake-executor-plugin-cluster-generic")' >/dev/null 2>&1; then
      python3 -m pip install snakemake-executor-plugin-cluster-generic==1.0.9 || return 1
    fi

    if [ "$setup_snakemake_target" == "snakemake_cern" ]; then
      setup_profile="profiles/cern"
      echo "Setting up CERN Condor for Snakemake"
      for setup_command in condor_submit condor_q condor_history condor_rm apptainer; do
        if ! command -v "$setup_command" >/dev/null 2>&1; then
          echo "Missing required command: $setup_command. Run snakemake_cern on lxplus with these tools available." >&2
          return 1
        fi
      done

    else
      echo "Setting up IC Condor for Snakemake"
      if ! python3 -c 'import htcondor' >/dev/null 2>&1; then
        python3 -m pip install htcondor || return 1
      fi
    fi

    python3 - "$setup_profile" <<'PY'
from pathlib import Path
import sys
import yaml

profile = Path(sys.argv[1])
site = profile.name.upper()
for name in ['config.yaml', 'condor.py', 'jobscript.sh']:
    if not (profile/name).is_file():
        sys.exit(f'Missing {site} profile file: {profile/name}. Run setup.sh from the repository root.')
settings = yaml.safe_load((profile/'config.yaml').read_text())
if not isinstance(settings, dict) or settings.get('executor') != 'cluster-generic':
    sys.exit(f'Invalid {site} profile: {profile}/config.yaml must use the cluster-generic executor.')
print(f'{site} profile ready: {profile.resolve()}')
PY
    if [ $? -ne 0 ]; then return 1; fi
    echo "Use --snakemake-profile=$setup_profile from the repository root."
  }

  # Keep the old target and no-argument setup working as IC setup.
  if ! _SetupSnakeMake "${1:-snakemake_ic}"; then
    return 1 2>/dev/null || exit 1
  fi
  unset -f _SetupSnakeMake
fi

if [ $# -eq 0 ] || [ "$1" == "container" ]; then
  echo "Building Apptainer image of the environment and current code"
  python3 scripts/build_apptainer.py \
    --env-prefix "${CONDA_PREFIX:-${MINICONDA_BASE:-./miniconda}/files/envs/innfer_env}" \
    --work-dir "${INNFER_CONTAINER_BUILD_DIR:-./tmp/apptainer-build}"
fi


#rm -rf tmp/*
