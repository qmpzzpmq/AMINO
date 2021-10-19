#!/bin/bash

conda create -y -n AMINO python=3.8
conda activate AMINO
conda install -y pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install -y -c conda-forge pytorch-lightning
conda install -y -c conda-forge hydra-core
pip install -r requirement.txt
echo "Install successful"