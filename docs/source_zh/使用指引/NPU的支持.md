# NPU（昇腾）开箱指南

本文档介绍如何在华为昇腾 NPU 环境下安装和使用 Twinkle 框架。

## 环境要求

在开始之前，请确保您的系统满足以下要求：

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | >= 3.11, < 3.13 | Twinkle 框架要求 |
| 昇腾固件驱动（HDK） | 推荐最新版本 | 硬件驱动和固件 |
| CANN 工具包 | 8.5.1 或更高 | 异构计算架构 |
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
- CANN: 8.5.1 或更高

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
pip install vllm==0.14.0

# 第二步：安装 vLLM-Ascend
pip install vllm-ascend==0.14.0rc1
```

**注意事项**：
- 按照上述顺序安装，忽略可能的依赖冲突提示
- 安装前确保已激活 CANN 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 推荐使用的版本为 vLLM 0.14.0 和 vLLM-Ascend 0.14.0rc1

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

### 5. 安装 Megatron 后端依赖

**推荐组合**：
- Megatron-LM: `v0.15.3`
- MindSpeed: `core_r0.15.3`
- mcore-bridge: 主分支或当前 Twinkle 验证过的版本

**安装步骤**：

```bash
# 1. 获取 Megatron-LM，并切到 Twinkle 兼容版本
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout v0.15.3
cd ..

# 2. 获取并安装 MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout core_r0.15.3
pip install -e .
cd ..

# 3. 获取并安装 mcore-bridge
git clone https://github.com/modelscope/mcore-bridge.git
cd mcore-bridge
pip install -e .
cd ..

# 4. 安装 Twinkle（如果还没有安装）
cd twinkle
pip install -e ".[transformers,ray]"
```

**运行前环境变量**：

```bash
export PYTHONPATH=$PYTHONPATH:<path/to/Megatron-LM>
export MEGATRON_LM_PATH=</path/to/Megatron-LM>
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

**验证方式**：

先跑一个最小导入检查，确认 MindSpeed / Megatron-LM 可以被当前环境找到：

```bash
python -c "import mindspeed.megatron_adaptor; from twinkle.model.megatron._mindspeed_runtime import ensure_mindspeed_adaptor_patched; ensure_mindspeed_adaptor_patched(); print('✓ Megatron backend imports are ready')"
```

## 快速开始

**重要提示**：以下示例均来自 `cookbook/` 目录，已在实际 NPU 环境中验证通过。建议直接运行 cookbook 中的脚本，而不是复制粘贴代码片段。

### SFT LoRA 微调

当前 NPU 文档不再提供这类 SFT cookbook 示例；这部分能力需要结合实际可用的 cookbook 示例或后续补充的 NPU 脚本来说明。

### GRPO 强化学习训练

当前 NPU 文档不再提供这类 GRPO cookbook 示例；这部分能力需要结合实际可用的 cookbook 示例或后续补充的 NPU 脚本来说明。

### 更多示例

查看 `cookbook/remote/tinker/ascend/` 目录了解远程训练服务端配置。

## 并行策略

Twinkle 在 NPU 上目前支持以下**经过验证**的并行策略：

| 并行类型 | 说明 | NPU 支持 | 验证状态 |
|---------|------|---------|---------|
| DP (Data Parallel) | 数据并行 | ✅ | 暂无对应 cookbook 示例 |
| FSDP (Fully Sharded Data Parallel) | 完全分片数据并行 | ✅ | 暂无对应 cookbook 示例 |
| TP (Tensor Parallel) | 张量并行（Megatron） | ✅ | 已验证（见 `cookbook/megatron/ascend/tp_npu.py`） |
| PP (Pipeline Parallel) | 流水线并行（Megatron） | ✅ | 已验证（见 `cookbook/megatron/ascend/tp_npu.py`） |
| CP (Context Parallel) | 上下文并行 | ✅ | 已验证（见 `cookbook/megatron/ascend/tp_moe_cp_npu.py`） |
| EP (Expert Parallel) | 专家并行（MoE） | ✅ | 已验证（见 `cookbook/megatron/ascend/tp_moe_npu.py`） |

**图例说明**：
- ✅ 已验证：有实际运行示例代码
- 🚧 待验证：理论上支持但暂无 NPU 验证示例
- ❌ 不支持：当前版本不可用

### DP + FSDP 示例

当前 NPU 文档暂不提供对应的 cookbook 代码片段。

**Megatron 后端说明**：Twinkle 的 Megatron NPU 路径已经提供了可直接运行的 smoke 示例，安装和运行依赖请参考上面的 “Megatron 后端依赖” 小节。当前优先建议先验证 `cookbook/megatron/ascend/tp_npu.py`，再逐步切到 `cookbook/megatron/ascend/tp_moe_npu.py` 和 `cookbook/megatron/ascend/tp_moe_cp_npu.py`。

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
| SFT + LoRA | ✅ | ✅ | - | 暂无对应 cookbook 示例 |
| GRPO | ✅ | ✅ | - | 暂无对应 cookbook 示例 |
| DP 并行 | ✅ | ✅ | - | 暂无对应 cookbook 示例 |
| FSDP 并行 | ✅ | ✅ | - | 暂无对应 cookbook 示例 |
| Ray 分布式 | ✅ | ✅ | - | 暂无对应 cookbook 示例 |
| TorchSampler | ✅ | ✅ | - | 暂无对应 cookbook 示例 |
| vLLMSampler | ✅ | ✅ | - | 暂无对应 cookbook 示例 |
| 全量微调 | ✅ | ✅ | - | 已验证可用 |
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
