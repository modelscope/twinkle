# syntax=docker/dockerfile:1.7

# ---------- Stage 0: grab pre-built binaries ----------
FROM redis:7.4.9 AS redis
FROM grafana/otel-lgtm:0.28.0 AS lgtm

# ---------- Stage 1: GPU training / serving image ----------
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.1-1.35.0

# Install miniconda with Python 3.12
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda create -n twinkle python=3.12 pip -y --override-channels -c conda-forge
ENV TWINKLE_PYTHON="/opt/conda/envs/twinkle/bin/python"
ENV PATH="/opt/conda/envs/twinkle/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/envs/twinkle/lib:${LD_LIBRARY_PATH}"
RUN ${TWINKLE_PYTHON} --version && ${TWINKLE_PYTHON} -m pip --version
RUN ${TWINKLE_PYTHON} -m pip config set global.index-url https://mirrors.cloud.aliyuncs.com/pypi/simple && \
    ${TWINKLE_PYTHON} -m pip config set install.trusted-host mirrors.cloud.aliyuncs.com

ENV SETUPTOOLS_USE_DISTUTILS=local

# Install vllm
RUN --mount=type=cache,target=/root/.cache/pip \
    ${TWINKLE_PYTHON} -m pip install vllm==0.19.1

# Install base packages
RUN --mount=type=cache,target=/root/.cache/pip \
    ${TWINKLE_PYTHON} -m pip install -U "peft<=0.18" accelerate transformers "modelscope[framework]"

# Install transformer_engine and megatron_core
RUN --mount=type=cache,target=/root/.cache/pip \
    SITE_PACKAGES=$(${TWINKLE_PYTHON} -c "import site; print(site.getsitepackages()[0])") && \
    CUDNN_PATH=$SITE_PACKAGES/nvidia/cudnn \
    CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/cudnn/include \
    ${TWINKLE_PYTHON} -m pip install --no-build-isolation "transformer_engine[pytorch]"

RUN --mount=type=cache,target=/root/.cache/pip \
    ${TWINKLE_PYTHON} -m pip install megatron_core mcore_bridge ms-swift==4.1.3

# Install flash-attention (default arch 8.0;9.0, override via build-arg if needed)
ARG TORCH_CUDA_ARCH_LIST="8.0;9.0"
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS=8 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    ${TWINKLE_PYTHON} -m pip install flash-attn --no-build-isolation

RUN --mount=type=cache,target=/root/.cache/pip \
    ${TWINKLE_PYTHON} -m pip install flash-linear-attention -U

RUN --mount=type=cache,target=/root/.cache/pip \
    ${TWINKLE_PYTHON} -m pip install tilelang==0.1.11 apache-tvm-ffi==0.1.11
RUN ${TWINKLE_PYTHON} -c "import tilelang"

# Install numpy
RUN --mount=type=cache,target=/root/.cache/pip \
    ${TWINKLE_PYTHON} -m pip install numpy==2.2

# Install tinker, ray, and other deps
RUN --mount=type=cache,target=/root/.cache/pip \
    ${TWINKLE_PYTHON} -m pip install tinker==0.16.1 "ray[serve]"

# Clone and install twinkle, checkout to latest v-tag
RUN git clone https://github.com/modelscope/twinkle.git
WORKDIR /twinkle
RUN echo "Available release branches:" && git branch -r -l 'origin/release/*' --sort=-v:refname && \
    LATEST_RELEASE=$(git branch -r -l 'origin/release/*' --sort=-v:refname | head -n 1 | tr -d ' ') && \
    echo "Checking out: $LATEST_RELEASE" && \
    git checkout --track "$LATEST_RELEASE"

# Install twinkle itself (with server extras: redis + otel + telemetry)
RUN --mount=type=cache,target=/root/.cache/pip \
    ${TWINKLE_PYTHON} -m pip install -e ".[server]" --no-build-isolation

# ---------- Redis server ----------
COPY --from=redis /usr/local/bin/redis-server /usr/local/bin/redis-server
COPY --from=redis /usr/local/bin/redis-cli /usr/local/bin/redis-cli

# ---------- Observability: Grafana LGTM stack ----------
COPY --from=lgtm /otel-lgtm /otel-lgtm
COPY cookbook/observability/grafana/dashboards/twinkle-overview.json \
     /otel-lgtm/grafana/conf/provisioning/dashboards/twinkle-overview.json
