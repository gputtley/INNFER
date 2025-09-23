mkdir -p tmp
export TMPDIR="./tmp/"
export TEMP=$TMPDIR
export TMP=$TMPDIR

if [ $# -eq 0 ] || [ "$1" == "conda" ]; then
  echo "Installing miniconda"
  mkdir miniconda
  pushd miniconda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh
  chmod +x Miniconda3-latest-Linux-x86\_64.sh 
  ./Miniconda3-latest-Linux-x86\_64.sh -b -p ./files
  source files/etc/profile.d/conda.sh
  conda update conda
  rm ./Miniconda3-latest-Linux-x86\_64.sh
  popd
  source miniconda/files/etc/profile.d/conda.sh
fi

if [ $# -eq 0 ] || [ "$1" == "env" ]; then
  echo "Creating enviroment"
  source miniconda/files/etc/profile.d/conda.sh
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
  source miniconda/files/etc/profile.d/conda.sh
  conda activate miniconda/files/envs/innfer_env
  pip3 install --user htcondor snakemake-executor-plugin-cluster-generic
  conda install -c conda-forge -c bioconda python-htcondor snakemake-executor-plugin-cluster-generic
  pip3 install cookiecutter
  mkdir ~/.config/snakemake
  pushd ~/.config/snakemake
  cookiecutter https://github.com/gputtley/htcondor.git
  popd
fi
