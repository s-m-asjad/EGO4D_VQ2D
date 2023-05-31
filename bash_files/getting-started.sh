#!/usr/bin/bash

# Download PyTracking

VQ2D_ROOT="/media/goku/4b66c306-b38b-4701-9bd5-fd5c65a905fd/asjad.s/EGO4D/vq2d_cvpr"

#source activate ego4d_vq2d

echo $VQ2D_ROOT
mkdir -p $VQ2D_ROOT/dependencies
cd $VQ2D_ROOT/dependencies
#export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
git clone https://github.com/visionml/pytracking
cd pytracking
#git checkout de9cb9bb4f8cad98604fe4b51383a1e66f1c45c0
git submodule update --init --recursive # Downloads the PreciseROIPooling library

# Install PyTracking dependencies
conda install matplotlib pandas tqdm --yes
pip install opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
conda install cython --yes
pip install pycocotools lvis

# Install ninja-build
#sudo apt-get install ninja-build
pip install ninja

# Build Pytorch-Correlation-Sampler
cd $VQ2D_ROOT/dependencies
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension
cd Pytorch-Correlation-extension
python setup.py install

# Other dependencies
pip install PyTurboJPEG
pip install jpeg4py av imageio-ffmpeg

# Initialize PyTracking Env Setting Files
cd $VQ2D_ROOT/dependencies/pytracking/

export LD_LIBRARY_PATH=/home/goku/anaconda3/envs/asjad/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

# Download pre-trained KYS tracker networks
mkdir $VQ2D_ROOT/pretrained_models/
cd $VQ2D_ROOT/pretrained_models/
gdown --id 1nJTBxpuBhN0WGSvG7Zm3yBc9JAC6LnEn
