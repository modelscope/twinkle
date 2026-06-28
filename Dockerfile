# ---------- Stage 0: grab pre-built binaries ----------
FROM redis:7.4.9 AS redis
FROM grafana/otel-lgtm:0.28.0 AS lgtm
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.21.0-modelscope1.36.3-swift4.2.3

# Put the real CUDA runtime ahead of package-private stubs, then keep the forward-compat
# driver shim and pip cuDNN ahead of apt cuDNN.
ENV LD_LIBRARY_PATH="/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/compat:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Only twinkle-specific deps; everything else (torch, vllm, TE, flash-attn, megatron-core, mcore-bridge, transformers, peft, accelerate) ships in the base.
RUN pip config set global.index-url https://mirrors.cloud.aliyuncs.com/pypi/simple && \
    pip config set install.trusted-host mirrors.cloud.aliyuncs.com && \
    pip install --no-cache-dir \
        tinker==0.16.1 \
        "ray[serve]" && \
    pip install --no-cache-dir --no-deps \
        flash-linear-attention \
        apache-tvm-ffi==0.1.9 \
        tilelang==0.1.9 && \
    ln -sf /usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so \
        /usr/local/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so && \
    python -c "import ctypes, os; stub_path = '/usr/local/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so'; lib = ctypes.CDLL(stub_path); getattr(lib, 'cudaDeviceReset'); print(f'tilelang cudart path: {os.path.realpath(stub_path)}')" && \
    rm -rf /root/.cache /tmp/*

# Clone and install twinkle, checkout to latest v-tag
RUN git clone https://github.com/modelscope/twinkle.git
WORKDIR /twinkle
RUN echo "Available release branches:" && git branch -r -l 'origin/release/*' --sort=-v:refname && \
    LATEST_RELEASE=$(git branch -r -l 'origin/release/*' --sort=-v:refname | head -n 1 | tr -d ' ') && \
    echo "Checking out: $LATEST_RELEASE" && \
    git checkout --track "$LATEST_RELEASE"

# Install twinkle itself (with server extras: redis + otel + telemetry)
RUN pip install -e ".[server]" --no-build-isolation

# ---------- Redis server ----------
COPY --from=redis /usr/local/bin/redis-server /usr/local/bin/redis-server
COPY --from=redis /usr/local/bin/redis-cli /usr/local/bin/redis-cli

# ---------- Observability: Grafana LGTM stack ----------
COPY --from=lgtm /otel-lgtm /otel-lgtm
COPY cookbook/observability/grafana/dashboards/twinkle-overview.json \
     /otel-lgtm/grafana/conf/provisioning/dashboards/twinkle-overview.json
