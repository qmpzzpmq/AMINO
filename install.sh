env_name=AMINO

conda env remove -n ${env_name}
conda create -y -n ${env_name} python=3.8
conda activate ${env_name}
# conda install -y pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install -n ${env_name} -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install -n ${env_name} -y -c conda-forge pytorch-lightning hydra-core wandb pysoundfile librosa
# conda install -n ${env_name} -y -c huggingface tokenizers transformersconda uninstall tokenizers, transformers
conda install -n ${env_name} -y -c anaconda h5py
pip install -r requirements.txt
echo "Install successful"
