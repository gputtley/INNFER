export TMPDIR="./tmp/"
mkdir miniconda
pushd miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh
chmod +x Miniconda3-latest-Linux-x86\_64.sh 
./Miniconda3-latest-Linux-x86\_64.sh -b -p ./files
source files/etc/profile.d/conda.sh
conda update conda
conda create --name innfer_env python=3.11
conda activate innfer_env
conda activate innfer_env
pip3 install bayesflow
pip3 install mplhep
pip3 install PyYAML
pip3 install pyarrow
popd