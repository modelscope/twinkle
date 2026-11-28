# ---------- Stage 0: grab pre-built binaries ----------
FROM redis:7.4.9 AS redis
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.14.0-vllm0.33.0-modelscope1.40.3-swift4.6.0@sha256:88edb21910b71f88924e3dea013b2dd383527adac9a3e32035cdeb574eb2e0b3

# Put the real CUDA runtime ahead of package-private stubs, then keep the forward-compat
# driver shim and pip cuDNN ahead of apt cuDNN.
ENV LD_LIBRARY_PATH="/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/compat:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /twinkle
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

COPY twinkle_cs/server/ ./twinkle_cs/server/

# Resolve the pinned Swift revision and Twinkle together. Keep the base image's
# CUDA wheels and let vLLM enforce the matching TileLang and TVM FFI versions.
RUN pip install --no-cache-dir --no-build-isolation \
    --only-binary=torch,vllm,ray,megatron-core,mcore-bridge \
    "torch==2.14.0" "vllm==0.33.0" \
    "ray[serve]==2.55.1" "megatron-core==0.20.0" "mcore-bridge==1.7.0" \
    "ms-swift[megatron] @ https://codeload.github.com/modelscope/ms-swift/tar.gz/e1ac89d24f47cb8fd73a7245dd79df0959aec284" \
    -e . -e ./twinkle_cs/server
COPY cookbook/ ./cookbook/
COPY cookbook/client/server/twinkle-runtime.sh /usr/local/bin/twinkle-runtime
COPY cookbook/client/server/twinkle-runtime-common.sh /usr/local/bin/

# ---------- Redis server ----------
COPY --from=redis /usr/local/bin/redis-server /usr/local/bin/redis-server
COPY --from=redis /usr/local/bin/redis-cli /usr/local/bin/redis-cli
