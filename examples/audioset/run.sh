#!/bin/bash

. ./AMINO/tools/path.sh
audioset_path=/audioset
num_line=$(wc -l ${audioset_path}/metadata/class_labels_indices.csv | cut -f 1 -d " ")
num_class=`expr ${num_line} - 1`

python AMINO/bin/train.py --config-name mel_classifier.yaml \
    ++variable.num_classes=${num_class}
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    hydra.run.dir=. \
    hydra.output_subdir=null \
