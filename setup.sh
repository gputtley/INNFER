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
  pip3 install bayesflow
  pip3 install mplhep
  pip3 install PyYAML
  pip3 install pyarrow
  pip3 install uproot
  pip3 install scipy
  pip3 install seaborn
  pip3 install scikit-learn
  pip3 install wandb
fi
