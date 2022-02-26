env_name=AMINO

conda env remove -n ${env_name} && \
conda create -y -n ${env_name} python=3.8 && \
conda install -n ${env_name} -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch && \
conda install -n ${env_name} -y -c conda-forge pytorch-lightning hydra-core wandb pysoundfile librosa torchmetrics && \
# conda install -n ${env_name} -y -c huggingface tokenizers transformersconda uninstall tokenizers, transformers
conda install -n ${env_name} -y -c anaconda h5py && \
source activate ${env_name} && \
conda run -n ${env_name} which python && \
conda run -n ${env_name} pip install -r requirements.txt && \
echo "conda install successful"
