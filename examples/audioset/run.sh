#!/bin/bash

set -x

num_enc_layer=9
limit_train_batches=1.0
limit_val_batches=1.0
batch_size=20
gpus="-1"
audioset_path="/share2/users/charlie/audioset/audioset_Kong"

. ./AMINO/tools/path.sh
. ./AMINO/tools/parse_options.sh || exit 1; # bash run.sh --node_rank 1

which python
for ((i = 0; i < ${gpus}; ++i)); do
{
    LOCAL_RANK=${i} python AMINO/bin/train.py --config-name HF_enc_classifier.yaml \
        hydra.job.name="HF_Num_layer${num_enc_layer}_Train${limit_train_batches}_Val${limit_val_batches}" \
        ++variables.batch_size=${batch_size} \
        ++variables.audio_path=${audioset_path} \
        ++trainer.auto_select_gpus=true \
        ++trainer.gpus=${gpus} \
        ++trainer.limit_train_batches=${limit_train_batches} \
        ++trainer.limit_val_batches=${limit_val_batches} \
        ++trainer.num_nodes=1 \
        ++module.conf.scheduler.conf.warmup_steps=30000 \
        ++module.conf.net.conf.encoder.conf.from_pretrained_num_hidden_layers=${num_enc_layer} \

} &
done
wait

