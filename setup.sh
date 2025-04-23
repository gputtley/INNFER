export TMPDIR="./tmp/"

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
  conda env create --file=configs/setup/environment.yaml
  pip3 install tensorflow==2.15.1 wrapt==1.14.1 --no-deps
  pip3 install snakemake snakemake-interface-storage-plugins wrapt --no-deps 
  conda activate miniconda/files/envs/innfer_env
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
