## Installation

### Requirements:
- PyTorch >= 1.3 (we recommend 1.8.1 for better speed)
- torchvision
- detectron2
- cocoapi
- cityscapesScripts
- apex
- GCC >= 4.9
- CUDA >= 9.2
- pip packages in [requirements.txt](requirements.txt)

### Step 1 (Option 1): Install torch with docker
```bash
# We recommend setting environment using nvidia-docker with nvcr.io docker image,
# where basic necessary packages (pytorch, torchvision, apex, CUDA, and others)
# are pre-installed.

# after installing nvidia-docker, we recommend using following docker image:
docker pull nvcr.io/nvidia/pytorch:21.02-py3

# run docker container. for details, please read page:
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
docker run --gpus all \
--name <your_container_name> \
--restart unless-stopped \
-it \
-v <dir_of_ImageNetVID>:/dataset \
--ipc=host \
-e TZ=<your time zone, ex. Asia/Seoul> \
nvcr.io/nvidia/pytorch:21.02-py3 \
/bin/bash

apt-get update
apt-get -y install libgl1-mesa-glx
```


### Step 1 (Option 2): Install torch with conda
```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do
# or you can use pytorch docker to easily setup the environment

conda create --name DAFA -y python=3.8
source activate DAFA

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.0
# conda install pytorch=1.3.0 torchvision cudatoolkit=10.0 -c pytorch
conda install pytorch=1.8.1 torchvision cudatoolkit=11.1 -c pytorch

sudo apt-get update
```

### Step 2: Package Installation
```bash
cd <your install dir>
export INSTALL_DIR=$PWD

# install detectron2:
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex (skip when you use nvcr docker)
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# clone our lib:
cd $INSTALL_DIR
git clone https://github.com/sdroh1027/DiffusionVID.git
cd DiffusionVID

# Then install pip packages:
pip install -r requirements.txt

# install the lib with symbolic links
python setup.py build develop

unset INSTALL_DIR

```
