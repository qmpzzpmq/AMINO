env_name=AMINO

conda env remove -n ${env_name} && \
conda create -y -n ${env_name} python=3.8 && \
source activate ${env_name} && \
conda run -n ${env_name} which python && \
conda run -n ${env_name} pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html && \
echo "install successful"
