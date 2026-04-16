FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.1-1.35.0

# Install miniconda with Python 3.12
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda create -n twinkle python=3.12 -y --override-channels -c conda-forge
ENV PATH="/opt/conda/envs/twinkle/bin:${PATH}"

ENV SETUPTOOLS_USE_DISTUTILS=local

# Install base packages
RUN pip install --upgrade peft accelerate transformers "modelscope[framework]" --no-cache-dir

# Install vllm
RUN pip install --upgrade vllm --no-cache-dir

# Install transformer_engine and megatron_core
RUN SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && \
    CUDNN_PATH=$SITE_PACKAGES/nvidia/cudnn \
    CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/cudnn/include \
    pip install --no-build-isolation "transformer_engine[pytorch]" --no-cache-dir

RUN pip install megatron_core mcore_bridge --no-cache-dir

# Install flash-attention (default arch 8.0;9.0, override via build-arg if needed)
ARG TORCH_CUDA_ARCH_LIST="8.0;9.0"
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS=8 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install flash-attn --no-build-isolation --no-cache-dir

RUN pip install flash-linear-attention -U --no-cache-dir

# Install numpy
RUN pip install numpy==2.2 --no-cache-dir

# Install tinker, ray, and other deps
RUN pip install --no-cache-dir tinker==0.16.1 "ray[serve]" transformers peft<=0.18 accelerate -U

# Clone and install twinkle, checkout to latest v-tag
RUN git clone https://github.com/modelscope/twinkle.git
WORKDIR /twinkle
RUN echo "Available release branches:" && git branch -r -l 'origin/release/*' --sort=-v:refname && \
    LATEST_RELEASE=$(git branch -r -l 'origin/release/*' --sort=-v:refname | head -n 1 | tr -d ' ') && \
    echo "Checking out: $LATEST_RELEASE" && \
    git checkout --track "$LATEST_RELEASE"

# Install twinkle itself
RUN pip install -e . --no-build-isolation
