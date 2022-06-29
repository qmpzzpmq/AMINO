#!/bin/bash

. ./AMINO/tools/path.sh

export HOST_NODE_ADDR=5871
python AMINO/bin/train.py \
    --config-name AEGMM.yaml \
    hydra.job.name=AEGMM \
    ++trainer.devices='[0]' \
    2>&1 | tee AEGMM.log

export HOST_NODE_ADDR=5872
python AMINO/bin/train.py \
    --config-name AE_GMM.yaml \
    hydra.job.name=AE_GMM \
    ++trainer.devices='[0]' \
    2>&1 | tee AE_GMM.log
