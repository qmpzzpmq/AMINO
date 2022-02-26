FROM 10.100.112.79:5000/pytorch/pytorch-inspur:21.07-py3

SHELL ["/bin/bash", "--login", "-c"]
USER root
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

WORKDIR /root
COPY ./install.sh ./install.sh
COPY ./requirements.txt ./requirements.txt

RUN \
    # . /opt/conda/etc/profile.d/conda.sh && \
    . ./install.sh && \
    conda clean -afy && \
    rm -rf ~/.cache/pip && \
    rm ./install.sh ./requirements.txt && \
    conda init bash && \
    cat ~/.bashrc && \
    echo "python done"
