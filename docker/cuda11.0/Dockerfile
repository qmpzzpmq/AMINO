FROM 10.100.112.79:5000/pytorch/pytorch:20.08-py3-cuda11
# FROM 10.100.112.79:5000/pytorch/pytorch-inspur:21.07-py3

SHELL ["/bin/bash", "--login", "-c"]
USER root
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore
ENV PATH="/root/miniconda3/bin:${PATH}"

WORKDIR /
COPY ./install.sh ./install.sh
COPY ./requirements.txt ./requirements.txt

RUN \
    wget -q  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    cat ~/.bashrc && \
    echo "export PATH='/root/miniconda3/bin:$PATH'" >> ~/.bashrc && \
    conda init bash && \
    cat ~/.bashrc && \
    echo "conda install done"

RUN \
    . /root/miniconda3/etc/profile.d/conda.sh && \
    . ./install.sh && \
    . /root/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate AMINO && \
    conda clean -afy && \
    rm -rf ~/.cache/pip && \
    rm ./install.sh ./requirements.txt && \
    echo "python done"

