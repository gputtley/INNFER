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
  conda create --name innfer_env python=3.11
  rm ./Miniconda3-latest-Linux-x86\_64.sh
  popd
  source miniconda/files/etc/profile.d/conda.sh
  conda activate miniconda/files/envs/innfer_env
fi

if [ $# -eq 0 ] || [ "$1" == "packages" ]; then
  echo "Installing packages"
  source miniconda/files/etc/profile.d/conda.sh
  conda activate miniconda/files/envs/innfer_env
  pip3 install snakemake
  pip3 install mplhep
  pip3 install PyYAML
  pip3 install pyarrow
  pip3 install uproot
  pip3 install scipy
  pip3 install seaborn
  pip3 install scikit-learn
  pip3 install wandb
  pip3 install pyfiglet
  pip3 install xgboost
  pip3 install random-word
  pip3 install optuna
  pip3 install bayesflow
  #conda env create -f configs/setup/environment.yaml
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