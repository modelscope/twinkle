#!/bin/bash

# Installation script - We offer a script to install the megatron and vllm related dependencies,
# which always occur error

set -e  # Exit immediately on error
export SETUPTOOLS_USE_DISTUTILS=local
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

# Install latest base packages
echo ""
echo "Installing peft, accelerate, transformers, modelscope..."
pip install --upgrade peft accelerate transformers "modelscope[framework]" --no-cache-dir

# Install latest vllm
echo ""
echo "Installing latest vllm..."
pip install --upgrade vllm --no-cache-dir

# Get site-packages path and install transformer_engine and megatron_core
echo ""
echo "Installing transformer_engine and megatron_core..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Site-packages path: $SITE_PACKAGES"

CUDNN_PATH=$SITE_PACKAGES/nvidia/cudnn \
CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/cudnn/include \
pip install --no-build-isolation "transformer_engine[pytorch]" --no-cache-dir

pip install megatron_core mcore_bridge --no-cache-dir

# Install flash-attention (force local build)
echo ""
echo "Installing flash-attention (local build for $GPU_NAME)..."
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
MAX_JOBS=8 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
pip install flash-attn --no-build-isolation --no-cache-dir

pip install flash-linear-attention -U --no-cache-dir

# Install numpy
echo ""
echo "Installing numpy==2.2 and deep_gemm..."
pip install numpy==2.2 --no-cache-dir

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
