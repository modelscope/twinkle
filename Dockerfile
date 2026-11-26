# ---------- Stage 0: grab pre-built binaries ----------
FROM redis:7.4.9 AS redis
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.21.0-modelscope1.36.3-swift4.2.3@sha256:34dcf45979d6be187c882f54645e82893d59f46568b5352403a8d9d2e18ed360

# Put the real CUDA runtime ahead of package-private stubs, then keep the forward-compat
# driver shim and pip cuDNN ahead of apt cuDNN.
ENV LD_LIBRARY_PATH="/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/compat:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Keep the CUDA stack from the base. Its Megatron Core 0.17 predates
# Twinkle and Swift's Megatron API requirement.
ARG PIP_INDEX_URL=https://mirrors.cloud.aliyuncs.com/pypi/simple
RUN pip install --no-cache-dir \
        --only-binary=ray,megatron-core,mcore-bridge \
        "ray[serve]==2.55.1" \
        "megatron-core==0.18.2" \
        "mcore-bridge==1.7.0" && \
    pip install --no-cache-dir --no-deps \
        flash-linear-attention \
        apache-tvm-ffi==0.1.9 \
        tilelang==0.1.9 && \
    ln -sf /usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so \
        /usr/local/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so && \
    python -c "import ctypes, os; stub_path = '/usr/local/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so'; lib = ctypes.CDLL(stub_path); getattr(lib, 'cudaDeviceReset'); print(f'tilelang cudart path: {os.path.realpath(stub_path)}')" && \
    rm -rf /root/.cache /tmp/*

WORKDIR /twinkle
COPY pyproject.toml setup.cfg README.md LICENSE ./
COPY src/ ./src/

# The base release predates swift.dev.template_contract. Use the same
# revision as the Twinkle client setup, with the base runtime dependencies.
RUN pip install --no-cache-dir --no-deps --no-build-isolation \
    "ms-swift @ https://codeload.github.com/modelscope/ms-swift/tar.gz/e1ac89d24f47cb8fd73a7245dd79df0959aec284"

# Install twinkle itself (with server extras: redis + otel + telemetry)
RUN pip install -e ".[server]" --no-build-isolation
COPY cookbook/ ./cookbook/
COPY cookbook/client/server/twinkle-runtime.sh /usr/local/bin/twinkle-runtime
COPY cookbook/client/server/twinkle-runtime-common.sh /usr/local/bin/

# ---------- Redis server ----------
COPY --from=redis /usr/local/bin/redis-server /usr/local/bin/redis-server
COPY --from=redis /usr/local/bin/redis-cli /usr/local/bin/redis-cli
