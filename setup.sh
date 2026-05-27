mkdir -p tmp
export TMPDIR="./tmp/"
export TEMP=$TMPDIR
export TMP=$TMPDIR
export CONDA_PKGS_DIRS=$TMPDIR
export PIP_CACHE_DIR=$TMPDIR
export XDG_CACHE_HOME=$TMPDIR


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
  source ${MINICONDA_BASE}/files/etc/profile.d/conda.sh
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
fi

if [ $# -eq 0 ] || [ "$1" == "snakemake_condor" ]; then
  echo "Setting up condor for snakemake"
  source ${MINICONDA_BASE}/files/etc/profile.d/conda.sh
  conda activate ${MINICONDA_BASE}/files/envs/innfer_env
  pip3 install --user htcondor snakemake-executor-plugin-cluster-generic
  conda install -c conda-forge -c bioconda python-htcondor snakemake-executor-plugin-cluster-generic
  pip3 install cookiecutter
  mkdir ~/.config/snakemake
  pushd ~/.config/snakemake
  cookiecutter https://github.com/gputtley/htcondor.git
  popd
fi

if [ $# -eq 0 ] || [ "$1" == "container" ]; then
  echo "Making container of conda environment"
  #conda install -c conda-forge conda-pack
  conda activate ${MINICONDA_BASE}/files/envs/innfer_env
  conda-pack -n ${MINICONDA_BASE}/files/envs/innfer_env -o innfer_env_container.tar.gz
fi


rm -rf tmp/*