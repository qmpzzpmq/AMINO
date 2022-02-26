#!/bin/bash

set -x

. ./AMINO/tools/path.sh
. /root/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate AMINO
which python
pip install -r ../../requirements.txt
num_enc_layer=3
dict=$(cat conf/HF_enc_classifier.yaml | shyaml get-value variables.audioset_path)/metadata/class_labels_indices.csv
num_class=$(wc -l ${dict})

export WANDB_API_KEY=6e042cf1f1fe7aaacda8327dc959172c4a0aa57a
python AMINO/bin/train.py --config-name HF_enc_classifier.yaml \
    hydra.job.name="HF_Num_layer-${num_enc_layer}_Train_1.0_Val_0.0" \
    ++variables.dataloaders_num_workers=10 \
    ++variables.batch_size=10 \
    ++trainer.auto_select_gpus=true \
    ++trainer.gpus=-1 \
    ++trainer.limit_train_batches=1.0 \
    ++trainer.limit_val_batches=0.0 \
    ++module.conf.scheduler.conf.warmup_steps=30000 \
    ++module.conf.net.conf.encoder.conf.from_pretrained_num_hidden_layers=${num_enc_layer} \

