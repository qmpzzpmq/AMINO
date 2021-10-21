#!/bin/bash

. ./path.sh

python AMINO/bin/train.py --config-name conf/default.yaml \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    hydra.run.dir=. \
    hydra.output_subdir=null \
