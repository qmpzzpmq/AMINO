#!/bin/bash

. ./AMINO/tools/path.sh

python AMINO/bin/train.py \
    --config-name AEGMM.yaml \
    hydra.job.name=AEGMM \
