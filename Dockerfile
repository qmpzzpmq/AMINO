FROM 10.100.112.79:5000/pytorch/pytorch:20.08-py3-cuda11

SHELL ["/bin/bash", "--login", "-c"]
USER root
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore
ENV PATH="/root/miniconda3/bin:${PATH}"

WORKDIR /
COPY conda_install.sh.sh /conda_install.sh.sh
COPY requirements.txt /requirements.txt

RUN \
    wget -q  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    . /root/miniconda3/etc/profile.d/conda.sh && \ 
    ./conda_install.sh.sh && \
    . /root/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate AMINO && \
    which python && \
    pip install -r requirements.txt && \
    conda clean -afy && \
    rm -rf ~/.cache/pip && \
    rm /conda_install.sh.sh ./requirements.txt && \
    echo "done"

