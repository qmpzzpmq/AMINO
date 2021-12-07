#!/bin/bash

conda create -y -n AMINO python=3.8
conda activate AMINO
# conda install -y pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install -y -c conda-forge pytorch-lightning hydra-core wandb
conda install -y -c anaconda h5py
pip install -r requirement.txt
echo "Install successful"
