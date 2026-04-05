FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.1-1.35.0

# Install miniconda with Python 3.12
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda create -n twinkle python=3.12 -y --override-channels -c conda-forge
SHELL ["conda", "run", "-n", "twinkle", "/bin/bash", "-c"]

# Clone and install twinkle, checkout to latest v-tag
RUN git clone https://github.com/modelscope/twinkle.git
WORKDIR /twinkle
RUN echo "Available v-tags:" && git tag -l 'v*' --sort=-v:refname && \
    LATEST_TAG=$(git tag -l 'v*' --sort=-v:refname | head -n 1) && \
    echo "Checking out: $LATEST_TAG" && \
    git checkout "$LATEST_TAG"

RUN sh INSTALL_MEGATRON.sh

RUN pip install --no-cache-dir tinker==0.14.0 "ray[serve]" transformers peft accelerate -U

RUN pip install -e . --no-build-isolation

CMD ["bash", "cookbook/client/server/megatron/run.sh"]
