export TMPDIR="./tmp/"
mkdir miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh
pushd miniconda
chmod +x Miniconda3-latest-Linux-x86\_64.sh 
./Miniconda3-latest-Linux-x86\_64.sh 
eval "$(./ shell.bash hook)" 
conda create --name innfer_env python=3.11
conda activate innfer_env
pip3 install bayesflow
pip3 install mplhep
pip3 install PyYAML
pip3 install pyarrow
popd