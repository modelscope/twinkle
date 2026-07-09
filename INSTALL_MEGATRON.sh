#!/bin/bash

# Installation script - We offer a script to install the megatron and vllm related dependencies,
# which always occur error

set -e  # Exit immediately on error
export SETUPTOOLS_USE_DISTUTILS=local
export UV_INDEX_URL=${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}
echo "=========================================="
echo "Starting deep learning dependencies installation..."
echo "=========================================="

# Detect GPU architecture from nvidia-smi
echo ""
echo "Detecting GPU architecture..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "Detected GPU: $GPU_NAME"

# Map GPU name to CUDA architecture
get_cuda_arch() {
    local gpu_name="$1"
    case "$gpu_name" in
        *H100*|*H200*|*H20*|*H800*)
            echo "9.0"
            ;;
        *A100*|*A800*|*A30*)
            echo "8.0"
            ;;
        *A10*|*A40*|*A16*|*A2*)
            echo "8.6"
            ;;
        *L40*|*L4*|*Ada*|*RTX\ 40*|*RTX\ 50*)
            echo "8.9"
            ;;
        *V100*)
            echo "7.0"
            ;;
        *T4*)
            echo "7.5"
            ;;
        *RTX\ 30*|*A6000*|*A5000*)
            echo "8.6"
            ;;
        *RTX\ 20*)
            echo "7.5"
            ;;
        *)
            echo "8.0;9.0"  # Default fallback
            ;;
    esac
}

TORCH_CUDA_ARCH_LIST=$(get_cuda_arch "$GPU_NAME")
export TORCH_CUDA_ARCH_LIST
echo "Using CUDA architecture: $TORCH_CUDA_ARCH_LIST"

# Install vllm 0.21.x (latest 0.2x uses CUDA 12 toolchain, avoids CUDA 13 CUTLASS conflicts)
echo ""
echo "Installing vllm 0.21..."
uv pip install "vllm>=0.21,<0.22"

# Install latest base packages
echo ""
echo "Installing peft, accelerate, transformers, modelscope..."
uv pip install --upgrade peft accelerate transformers "modelscope[framework]"

# Get site-packages path and install transformer_engine and megatron_core
echo ""
echo "Installing transformer_engine and megatron_core..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Site-packages path: $SITE_PACKAGES"

export CUDA_HOME=${SITE_PACKAGES}/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
uv pip install transformer_engine_torch --no-build-isolation

uv pip install megatron_core mcore_bridge

# Install flash-attention
# Prefer prebuilt wheel; fall back to source build only if needed.
echo ""
echo "Installing flash-attention..."
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS=8
pip install flash-attn --no-cache-dir || \
    FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation --no-cache-dir

uv pip install flash-linear-attention --upgrade

# Install numpy
echo ""
echo "Installing numpy==2.2 and deep_gemm..."
uv pip install numpy==2.2

# Verify installation
echo ""
echo "Verifying installation..."
echo ""
python -c "
import pkg_resources

packages = ['peft', 'accelerate', 'transformers', 'modelscope', 'vllm', 'transformer_engine', 'megatron_core', 'flash_attn', 'numpy']

print('Installed package versions:')
print('-' * 40)
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'{pkg}: {version}')
    except pkg_resources.DistributionNotFound:
        print(f'{pkg}: Not installed')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
