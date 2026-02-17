# NPU (Ascend) Quick Start Guide

This document describes how to install and use the Twinkle framework in Huawei Ascend NPU environments.

## Environment Requirements

Before getting started, please ensure your system meets the following requirements:

| Component                    | Version Requirement        | Description                          |
|------------------------------|----------------------------|--------------------------------------|
| Python                       | >= 3.11, < 3.13            | Twinkle framework requirement        |
| Ascend Firmware Driver (HDK) | Latest version recommended | Hardware driver and firmware         |
| CANN Toolkit                 | 8.3.RC1 or higher          | Heterogeneous Computing Architecture |
| PyTorch                      | 2.7.1                      | Deep learning framework              |
| torch_npu                    | 2.7.1                      | Ascend PyTorch adapter plugin        |

**Important Notes**:
- torch and torch_npu versions **must be exactly the same** (e.g., both 2.7.1)
- Python 3.11 is recommended for best compatibility
- CANN toolkit requires approximately 10GB+ disk space

## Supported Hardware

Twinkle currently supports the following Ascend NPU devices:

- Ascend 910 series
- Other compatible Ascend accelerator cards

## Installation Steps

### 1. Install NPU Environment (Driver, CANN, torch_npu)

NPU environment installation includes Ascend driver, CANN toolkit, PyTorch, and torch_npu.

**📖 Complete Installation Tutorial**: [torch_npu Official Installation Guide](https://gitcode.com/Ascend/pytorch/overview)

This documentation includes:
- Ascend driver (HDK) installation steps
- CANN toolkit installation steps
- PyTorch and torch_npu installation steps
- Version compatibility instructions

**Recommended Version Configuration**:
- Python: 3.11
- PyTorch: 2.7.1
- torch_npu: 2.7.1
- CANN: 8.3.RC1 or higher

### 2. Install Twinkle

After NPU environment configuration is complete, install the Twinkle framework from source:

```bash
git clone https://github.com/modelscope/twinkle.git
cd twinkle
pip install -e ".[transformers,ray]"
```

### 3. Install vLLM and vLLM-Ascend (Optional)

If you need to use vLLMSampler for efficient inference, you can install vLLM and vLLM-Ascend.

**Installation Steps**:

```bash
# Step 1: Install vLLM
pip install vllm==0.11.0

# Step 2: Install vLLM-Ascend
pip install vllm-ascend==0.11.0rc3
```

**Notes**:
- Install in the above order, ignoring possible dependency conflict warnings
- Ensure CANN environment is activated before installation: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- Recommended versions are vLLM 0.11.0 and vLLM-Ascend 0.11.0rc3

### 4. Verify Installation

Create test script `verify_npu.py`:

```python
import torch
import torch_npu

print(f"PyTorch version: {torch.__version__}")
print(f"torch_npu version: {torch_npu.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU device count: {torch.npu.device_count()}")

if torch.npu.is_available():
    print(f"Current NPU device: {torch.npu.current_device()}")
    print(f"NPU device name: {torch.npu.get_device_name(0)}")

    # Simple test
    x = torch.randn(3, 3).npu()
    y = torch.randn(3, 3).npu()
    z = x + y
    print(f"NPU computation test passed: {z.shape}")
```

Run verification:

```bash
python verify_npu.py
```

If the output shows `NPU available: True` and no errors, installation is successful!

**Note**: Twinkle does not currently provide NPU Docker images. Manual installation is recommended. For containerized deployment, please refer to official images from the Ascend community.

## Quick Start

**Important Notice**: The following examples are from the `cookbook/` directory and have been verified in actual NPU environments. It is recommended to run scripts directly from the cookbook rather than copying and pasting code snippets.

### SFT LoRA Fine-tuning

Verified 4-card DP+FSDP training example:

**Example Path**: [cookbook/sft/lora_npu.py](https://github.com/modelscope/twinkle/blob/main/cookbook/sft/lora_npu.py)

**Run Method**:
```bash
# Specify using 4 NPU cards
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# Run training
python cookbook/sft/lora_npu.py
```

**Example Features**:
- ✅ Ray distributed mode
- ✅ DP + FSDP hybrid parallelism (2x2)
- ✅ LoRA fine-tuning
- ✅ Complete data loading and training loop

### GRPO Reinforcement Learning Training

Verified multi-card GRPO training example:

**Example Path**: [cookbook/grpo/lora_npu.py](https://github.com/modelscope/twinkle/blob/main/cookbook/grpo/lora_npu.py)

**Run Method**:
```bash
# Specify using 8 NPU cards
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training
python cookbook/grpo/lora_npu.py
```

**Example Features**:
- ✅ Actor-Critic architecture
- ✅ Supports Reference Model
- ✅ Optional TorchSampler or vLLMSampler
- ✅ Complete RL training workflow

### More Examples

Check the `cookbook/remote/tinker/ascend/` directory for remote training server-side configuration.

## Parallelization Strategies

Twinkle currently supports the following **verified** parallelization strategies on NPU:

| Parallel Type | Description | NPU Support | Verification Status |
|---------|------|---------|---------|
| DP (Data Parallel) | Data parallelism | ✅ | Verified (see cookbook/sft/lora_npu.py) |
| FSDP (Fully Sharded Data Parallel) | Fully sharded data parallelism | ✅ | Verified (see cookbook/sft/lora_npu.py) |
| TP (Tensor Parallel) | Tensor parallelism (Megatron) | 🚧 | To be verified |
| PP (Pipeline Parallel) | Pipeline parallelism (Megatron) | 🚧 | To be verified |
| CP (Context Parallel) | Context parallelism | 🚧 | To be verified |
| EP (Expert Parallel) | Expert parallelism (MoE) | 🚧 | To be verified |

**Legend**:
- ✅ Verified: Has actual running example code
- 🚧 To be verified: Theoretically supported but no NPU verification example yet
- ❌ Not supported: Not available in current version

### DP + FSDP Example

The following example is from `cookbook/sft/lora_npu.py`, verified in actual NPU environment:

```python
import numpy as np
from twinkle import DeviceMesh

# 4 cards: DP=2, FSDP=2
device_mesh = DeviceMesh(
    device_type='npu',
    mesh=np.array([[0, 1], [2, 3]]),
    mesh_dim_names=('dp', 'fsdp')
)
```

**Note**: Megatron backend (TP/PP/EP) support on NPU is under development, with no available examples yet. If you need these advanced parallelization strategies, please verify in GPU environment first or follow project updates.

## Common Issues

### 1. torch_npu Version Mismatch

**Problem**: Version incompatibility warnings or errors after installing torch_npu.

**Solution**:
- Ensure torch and torch_npu versions are exactly the same
- Check if CANN version is compatible with torch_npu

```bash
# Check current versions
python -c "import torch; import torch_npu; print(torch.__version__, torch_npu.__version__)"

# Reinstall matching versions
pip uninstall torch torch_npu -y
pip install torch==2.7.1
pip install torch_npu-2.7.1-cp311-cp311-linux_aarch64.whl
```

### 2. CANN Toolkit Version Issue

**Problem**: CANN version incompatible with torch_npu.

**Solution**:
- Refer to [Ascend Community Version Compatibility Table](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/softwareinstall/instg/atlasdeploy_03_0015.html)
- Install corresponding CANN toolkit version

## Feature Support Status

Feature support matrix based on actual code verification:

| Feature | GPU | NPU | Verification Example | Description |
|------|-----|-----|---------|------|
| SFT + LoRA | ✅ | ✅ | cookbook/sft/lora_npu.py | Verified available |
| GRPO | ✅ | ✅ | cookbook/grpo/lora_npu.py | Verified available |
| DP Parallelism | ✅ | ✅ | cookbook/sft/lora_npu.py | Verified available |
| FSDP Parallelism | ✅ | ✅ | cookbook/sft/lora_npu.py | Verified available |
| Ray Distributed | ✅ | ✅ | cookbook/sft/lora_npu.py | Verified available |
| TorchSampler | ✅ | ✅ | cookbook/grpo/lora_npu.py | Verified available |
| vLLMSampler | ✅ | ✅ | cookbook/grpo/lora_npu.py | Verified available |
| Full Fine-tuning | ✅ | 🚧 | - | Theoretically supported, to be verified |
| QLoRA | ✅ | ❌ | - | Quantization operators not yet supported |
| DPO | ✅ | 🚧 | - | Theoretically supported, to be verified |
| Megatron TP/PP | ✅ | 🚧 | - | To be adapted and verified |
| Flash Attention | ✅ | ⚠️ | - | Some operators not supported |

**Legend**:
- ✅ **Verified**: Has actual running example, confirmed available
- 🚧 **To be verified**: Theoretically supported but no NPU environment verification yet
- ⚠️ **Partial support**: Available but with limitations or performance differences
- ❌ **Not supported**: Not available in current version

**Usage Recommendations**:
1. Prioritize features marked as "Verified" for guaranteed stability
2. "To be verified" features can be attempted but may encounter compatibility issues
3. Refer to corresponding example code when encountering problems

## Example Code

Twinkle provides the following verified NPU training examples:

### SFT Training
- **4-card DP+FSDP LoRA Fine-tuning**: [cookbook/sft/lora_npu.py](https://github.com/modelscope/twinkle/blob/main/cookbook/sft/lora_npu.py)
  - Uses Ray mode for distributed training
  - Demonstrates DP + FSDP hybrid parallelism
  - Includes complete data loading and training loop

### GRPO Training
- **Multi-card GRPO RL Training**: [cookbook/grpo/lora_npu.py](https://github.com/modelscope/twinkle/blob/main/cookbook/grpo/lora_npu.py)
  - Actor-Critic architecture
  - Supports Reference Model
  - Optional TorchSampler or vLLMSampler

### Remote Training (Tinker Protocol)
- **Server Configuration**: [cookbook/remote/tinker/ascend/](https://github.com/modelscope/twinkle/tree/main/cookbook/remote/tinker/ascend)
  - Provides HTTP API interface
  - Supports remote training and inference
  - Suitable for production environment deployment

**Running Examples**:
```bash
# SFT training
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python cookbook/sft/lora_npu.py

# GRPO training
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python cookbook/grpo/lora_npu.py
```

## Reference Resources

- [Ascend Community Official Website](https://www.hiascend.com/)
- [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/softwareinstall/instg/atlasdeploy_03_0001.html)
- [torch_npu GitHub](https://github.com/Ascend/pytorch)
- [Twinkle GitHub](https://github.com/modelscope/twinkle)
- [Twinkle Documentation](https://twinkle.readthedocs.io/)

## Getting Help

If you encounter issues during use:

1. **Check Logs**: Set environment variable `ASCEND_GLOBAL_LOG_LEVEL=1` for detailed logs
2. **Submit Issue**: [Twinkle GitHub Issues](https://github.com/modelscope/twinkle/issues)
3. **Community Discussion**: [Ascend Community Forum](https://www.hiascend.com/forum)

## Next Steps

- 📖 Read [Quick Start](Quick-Start.md) for more training examples
- 📖 Read [Installation Guide](Installation.md) for other platform installations
- 🚀 Browse the `cookbook/` directory for complete example code
- 💡 Check [Twinkle Documentation](https://twinkle.readthedocs.io/) for advanced features
