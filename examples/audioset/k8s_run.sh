#!/bin/bash --login


set -x
cd `dirname $0` 

audioset_path="/audioset_Kong"
gpus="4"

. AMINO/tools/parse_options.sh || exit 1; # bash run.sh --node_rank 1

source activate ../../docker/DC/conda_env/AMINO

echo master_port: $MASTER_PORT
echo rank: $RANK
echo node_rank: $NODE_RANK
echo local_rank: $LOCAL_RANK
echo world_size: $WORLD_SIZE

# python ./AMINO/tools/running_test.py
./run.sh \
    --audioset_path ${audioset_path} \
    --gpus ${gpus} \

