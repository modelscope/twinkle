# NPU（昇腾）开箱指南

本文档介绍如何在华为昇腾 NPU 环境下安装和使用 Twinkle 框架。

## 环境要求

在开始之前，请确保您的系统满足以下要求：

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | >= 3.11, < 3.13 | Twinkle 框架要求 |
| 昇腾固件驱动（HDK） | 推荐最新版本 | 硬件驱动和固件 |
| CANN 工具包 | 8.3.RC1 或更高 | 异构计算架构 |
| PyTorch | 2.7.1 | 深度学习框架 |
| torch_npu | 2.7.1 | 昇腾 PyTorch 适配插件 |

**重要说明**：
- torch 和 torch_npu 版本**必须完全一致**（例如都为 2.7.1）
- 推荐使用 Python 3.11 以获得最佳兼容性
- CANN 工具包需要约 10GB+ 磁盘空间

## 支持的硬件

Twinkle 当前支持以下昇腾 NPU 设备：

- 昇腾 910 系列
- 其他兼容的昇腾加速卡

## 安装步骤

### 1. 安装 NPU 环境（驱动、CANN、torch_npu）

NPU 环境的安装包括昇腾驱动、CANN 工具包、PyTorch 和 torch_npu。

**📖 完整安装教程**：[torch_npu 官方安装指南](https://gitcode.com/Ascend/pytorch/overview)

该文档包含：
- 昇腾驱动（HDK）安装步骤
- CANN 工具包安装步骤
- PyTorch 和 torch_npu 安装步骤
- 版本配套说明

**推荐版本配置**：
- Python: 3.11
- PyTorch: 2.7.1
- torch_npu: 2.7.1
- CANN: 8.3.RC1 或更高

### 2. 安装 Twinkle

NPU 环境配置完成后，从源码安装 Twinkle 框架：

```bash
git clone https://github.com/modelscope/twinkle.git
cd twinkle
pip install -e ".[transformers,ray]"
```

### 3. 安装 vLLM 和 vLLM-Ascend（可选）

如果需要使用 vLLMSampler 进行高效推理，可以安装 vLLM 和 vLLM-Ascend。

**安装步骤**：

```bash
# 第一步：安装 vLLM
pip install vllm==0.11.0

# 第二步：安装 vLLM-Ascend
pip install vllm-ascend==0.11.0rc3
```

**注意事项**：
- 按照上述顺序安装，忽略可能的依赖冲突提示
- 安装前确保已激活 CANN 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 推荐使用的版本为 vLLM 0.11.0 和 vLLM-Ascend 0.11.0rc3

### 4. 验证安装

创建测试脚本 `verify_npu.py`：

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

    # 简单测试
    x = torch.randn(3, 3).npu()
    y = torch.randn(3, 3).npu()
    z = x + y
    print(f"NPU computation test passed: {z.shape}")
```

运行验证：

```bash
python verify_npu.py
```

如果输出显示 `NPU available: True` 且没有报错，说明安装成功！

**注意**：目前 Twinkle 暂未提供 NPU 的 Docker 镜像，建议使用手动安装方式。如需容器化部署，请参考昇腾社区的官方镜像。

## 快速开始

**重要提示**：以下示例均来自 `cookbook/` 目录，已在实际 NPU 环境中验证通过。建议直接运行 cookbook 中的脚本，而不是复制粘贴代码片段。

### SFT LoRA 微调

已验证的 4 卡 DP+FSDP 训练示例：

**示例路径**：[cookbook/sft/lora_npu.py](https://github.com/modelscope/twinkle/blob/main/cookbook/sft/lora_npu.py)

**运行方式**：
```bash
# 指定使用 4 张 NPU 卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# 运行训练
python cookbook/sft/lora_npu.py
```

**示例特性**：
- ✅ Ray 分布式模式
- ✅ DP + FSDP 混合并行（2x2）
- ✅ LoRA 微调
- ✅ 完整的数据加载和训练循环

### GRPO 强化学习训练

已验证的多卡 GRPO 训练示例：

**示例路径**：[cookbook/grpo/lora_npu.py](https://github.com/modelscope/twinkle/blob/main/cookbook/grpo/lora_npu.py)

**运行方式**：
```bash
# 指定使用 8 张 NPU 卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 运行训练
python cookbook/grpo/lora_npu.py
```

**示例特性**：
- ✅ Actor-Critic 架构
- ✅ 支持 Reference Model
- ✅ 可选 TorchSampler 或 vLLMSampler
- ✅ 完整的 RL 训练流程

### 更多示例

查看 `cookbook/remote/tinker/ascend/` 目录了解远程训练服务端配置。

## 并行策略

Twinkle 在 NPU 上目前支持以下**经过验证**的并行策略：

| 并行类型 | 说明 | NPU 支持 | 验证状态 |
|---------|------|---------|---------|
| DP (Data Parallel) | 数据并行 | ✅ | 已验证（见 cookbook/sft/lora_npu.py） |
| FSDP (Fully Sharded Data Parallel) | 完全分片数据并行 | ✅ | 已验证（见 cookbook/sft/lora_npu.py） |
| TP (Tensor Parallel) | 张量并行（Megatron） | 🚧 | 待验证 |
| PP (Pipeline Parallel) | 流水线并行（Megatron） | 🚧 | 待验证 |
| CP (Context Parallel) | 上下文并行 | 🚧 | 待验证 |
| EP (Expert Parallel) | 专家并行（MoE） | 🚧 | 待验证 |

**图例说明**：
- ✅ 已验证：有实际运行示例代码
- 🚧 待验证：理论上支持但暂无 NPU 验证示例
- ❌ 不支持：当前版本不可用

### DP + FSDP 示例

以下示例来自 `cookbook/sft/lora_npu.py`，在实际 NPU 环境中验证通过：

```python
import numpy as np
from twinkle import DeviceMesh

# 4 卡：DP=2, FSDP=2
device_mesh = DeviceMesh(
    device_type='npu',
    mesh=np.array([[0, 1], [2, 3]]),
    mesh_dim_names=('dp', 'fsdp')
)
```

**注意**：Megatron 后端（TP/PP/EP）在 NPU 上的支持正在开发中，暂无可用示例。如需使用这些高级并行策略，请先在 GPU 环境下验证，或关注项目更新。

## 常见问题

### 1. torch_npu 版本不匹配

**问题**：安装 torch_npu 后出现版本不兼容警告或错误。

**解决方案**：
- 确保 torch 和 torch_npu 版本完全一致
- 检查 CANN 版本是否与 torch_npu 兼容

```bash
# 查看当前版本
python -c "import torch; import torch_npu; print(torch.__version__, torch_npu.__version__)"

# 重新安装匹配版本
pip uninstall torch torch_npu -y
pip install torch==2.7.1
pip install torch_npu-2.7.1-cp311-cp311-linux_aarch64.whl
```

### 2. CANN 工具包版本问题

**问题**：CANN 版本与 torch_npu 不兼容。

**解决方案**：
- 参考[昇腾社区版本配套表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/softwareinstall/instg/atlasdeploy_03_0015.html)
- 安装对应版本的 CANN 工具包

## 功能支持情况

基于实际代码验证的功能支持矩阵：

| 功能 | GPU | NPU | 验证示例 | 说明 |
|------|-----|-----|---------|------|
| SFT + LoRA | ✅ | ✅ | cookbook/sft/lora_npu.py | 已验证可用 |
| GRPO | ✅ | ✅ | cookbook/grpo/lora_npu.py | 已验证可用 |
| DP 并行 | ✅ | ✅ | cookbook/sft/lora_npu.py | 已验证可用 |
| FSDP 并行 | ✅ | ✅ | cookbook/sft/lora_npu.py | 已验证可用 |
| Ray 分布式 | ✅ | ✅ | cookbook/sft/lora_npu.py | 已验证可用 |
| TorchSampler | ✅ | ✅ | cookbook/grpo/lora_npu.py | 已验证可用 |
| vLLMSampler | ✅ | ✅ | cookbook/grpo/lora_npu.py | 已验证可用 |
| 全量微调 | ✅ | 🚧 | - | 理论支持，待验证 |
| QLoRA | ✅ | ❌ | - | 量化算子暂不支持 |
| DPO | ✅ | 🚧 | - | 理论支持，待验证 |
| Megatron TP/PP | ✅ | 🚧 | - | 待适配和验证 |
| Flash Attention | ✅ | ⚠️ | - | 部分算子不支持 |

**图例说明**：
- ✅ **已验证**：有实际运行示例，确认可用
- 🚧 **待验证**：理论上支持但暂无 NPU 环境验证
- ⚠️ **部分支持**：可用但有限制或性能差异
- ❌ **不支持**：当前版本不可用

**使用建议**：
1. 优先使用标记为“已验证”的功能，稳定性有保障
2. “待验证”功能可以尝试，但可能遇到兼容性问题
3. 遇到问题时，参考对应的示例代码进行配置

## 示例代码

Twinkle 提供了以下经过验证的 NPU 训练示例：

### SFT 训练
- **4 卡 DP+FSDP LoRA 微调**：[cookbook/sft/lora_npu.py](https://github.com/modelscope/twinkle/blob/main/cookbook/sft/lora_npu.py)
  - 使用 Ray 模式进行分布式训练
  - 演示 DP + FSDP 混合并行
  - 包含完整的数据加载和训练循环

### GRPO 训练
- **多卡 GRPO RL 训练**：[cookbook/grpo/lora_npu.py](https://github.com/modelscope/twinkle/blob/main/cookbook/grpo/lora_npu.py)
  - Actor-Critic 架构
  - 支持参考模型（Reference Model）
  - 可选 TorchSampler 或 vLLMSampler

### 远程训练（Tinker 协议）
- **服务端配置**：[cookbook/remote/tinker/ascend/](https://github.com/modelscope/twinkle/tree/main/cookbook/remote/tinker/ascend)
  - 提供 HTTP API 接口
  - 支持远程训练和推理
  - 适用于生产环境部署

**运行示例**：
```bash
# SFT 训练
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python cookbook/sft/lora_npu.py

# GRPO 训练
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python cookbook/grpo/lora_npu.py
```

## 参考资源

- [昇腾社区官网](https://www.hiascend.com/)
- [CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/softwareinstall/instg/atlasdeploy_03_0001.html)
- [torch_npu GitHub](https://github.com/Ascend/pytorch)
- [Twinkle GitHub](https://github.com/modelscope/twinkle)
- [Twinkle 文档](https://twinkle.readthedocs.io/)

## 获取帮助

如果您在使用过程中遇到问题：

1. **查看日志**：设置环境变量 `ASCEND_GLOBAL_LOG_LEVEL=1` 获取详细日志
2. **提交 Issue**：[Twinkle GitHub Issues](https://github.com/modelscope/twinkle/issues)
3. **社区讨论**：[昇腾社区论坛](https://www.hiascend.com/forum)

## 下一步

- 📖 阅读 [快速开始](Quick-start.md) 了解更多训练示例
- 📖 阅读 [安装指南](Installation.md) 了解其他平台的安装
- 🚀 浏览 `cookbook/` 目录查看完整示例代码
- 💡 查看 [Twinkle 文档](https://twinkle.readthedocs.io/) 了解高级功能
