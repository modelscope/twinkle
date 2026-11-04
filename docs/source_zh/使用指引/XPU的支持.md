# XPU（昆仑芯）开箱指南

本文档介绍如何在昆仑芯 XPU（P800 系列）环境下安装和使用 Twinkle 框架。

## XPU 支持原理

与昇腾 NPU 不同，昆仑芯 XPU 在 Twinkle 中**不引入独立的设备类型**，而是依赖 `torch_xmlir` 在 import 阶段对 `torch.cuda` 符号进行重写：

- `torch.cuda.is_available()` 返回 `True`，设备表现为 `cuda:N`
- `libbkcl.so`（BKCL）以 `nccl` 后端（XCCL）挂载到 PyTorch
- `CUDA_VISIBLE_DEVICES` 由 XPU runtime 接管

得益于这种 "cuda-alike" 路线，Twinkle 的 GPU 代码路径绝大部分可原生复用，仅需适配少数符号重写未覆盖的 CUDA 专属假设。因此 `XPU` 平台**继承自 `GPU`**（`src/twinkle/utils/platforms/xpu.py`），而非像 NPU 那样是完全独立的设备后端。

## 环境要求

适配在厂商容器镜像（`kylin_v11-vllm_021_swift:*`）内验证，已确认的组件矩阵如下：

| 组件 | 版本 | 说明 |
|---|---|---|
| OS / Python | Kylin V11 / 3.10.14 | 满足 Twinkle `>=3.10` |
| torch | 2.9.0 | 被 `torch_xmlir` 符号重写 |
| vllm / vllm_kunlun | 0.21.0 / 0.21.0.dev0（0.25.1 亦复验通过） | `KunlunPlatform: device_type="cuda", dist_backend="nccl"` |
| megatron-core / transformers / triton | 0.16.1 / 5.9.0 / 3.5.0 | triton 走 xmlir 后端 |
| XPU kernels | xpu_flash_attn、xpu_fla、kunlun_ops、xspeedgate_ops、xformers | 由厂商容器提供 |

**说明**：
- 昆仑芯驱动、`torch_xmlir` 以及 XPU 版的 `torch` / `vllm_kunlun` 由昆仑芯提供（通常通过厂商容器镜像），Twinkle 不负责安装。
- `vllm_kunlun` 及其依赖来自昆仑芯；部分算子当前存在 dtype 限制（见[已知限制](#已知限制)）。

## 支持的硬件

- 昆仑芯 P800（OAM），已在单机 8 卡环境验证

## 安装步骤

### 1. 准备 XPU 环境

使用昆仑芯提供的容器镜像（或已安装昆仑芯驱动、`torch_xmlir` 及 XPU 版 `torch` / `vllm` / `vllm_kunlun` 的宿主机）。这些组件不由 Twinkle 管理。

### 2. 安装 Twinkle

从源码以 `--no-deps` 方式安装，避免 pip 覆盖厂商的 `torch` / `vllm` 构建：

```bash
git clone https://github.com/modelscope/twinkle.git
cd twinkle
pip install -e . --no-deps
pip install pyzmq
# 若容器缺少调试/运行时辅助依赖：
pip install h5py prettytable func_timeout ray redis
```

### 3. 验证安装

创建测试脚本 `verify_xpu.py`：

```python
import torch
import twinkle  # 触发 ensure_xpu_compat()
from twinkle.utils.platforms import Platform

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA (XPU) available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Detected platform: {Platform.get_platform().__name__}")

if torch.cuda.is_available():
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    print(f"XPU computation test passed: {(x + y).shape}")
```

运行验证：

```bash
python verify_xpu.py
```

运行成功时应输出 `Detected platform: XPU`、`CUDA (XPU) available: True` 以及正确的设备数量。平台检测依据 `PATH` 中是否存在 `xpu-smi`。

## 已验证能力

以下能力已在昆仑芯 P800（单机 8 卡）上端到端验证：

| 能力 | 后端 | 状态 | 说明 |
|---|---|---|---|
| FSDP2 LoRA SFT | transformers | ✅ 已验证 | 单卡，loss 收敛与 GPU 一致 |
| vLLM 采样（TP=1） | vLLM | ✅ 已验证 | 需 `enforce_eager=True`；纯文本模型 |
| vLLM 采样（TP=2） | vLLM | ✅ 已验证 | 张量并行 |
| GRPO（colocate） | native_fsdp | ✅ 已验证 | 全量权重同步（`merge_and_sync=True`）；relay ~0.7 GB/s |
| Megatron LoRA（TP=2） | Megatron | ✅ 已验证 | Twinkle 侧零改动；cuda-alike 路径 |
| 权重同步（训练↔采样） | XCCL | ✅ 已验证 | 经 `XCCLCheckpointEngine`，单机 relay |

## 已知限制

以下为当前昆仑芯 XPU 栈（厂商 kernel / `vllm_kunlun`）的限制，而非 Twinkle 逻辑问题：

- **LoRA 采样不可用**：`enable_lora=True` 的 `vLLMSampler` 无法启动。底层 LoRA 算子（`bgmv_shrink_cluster`、`sgmv_expand_sdnn`、`sgmv_expand_slice`）在 0.21 栈上仅支持 fp16。RL 请使用全量权重同步（`merge_and_sync=True`）。（0.25.1 栈厂商新增了 bf16 适配层，但 LoRA 路径仍有其他问题。）
- **部分模型需 fp16**：GDN FLA 算子仅支持 fp16，Qwen3.5 推理必须用 float16。
- **CUDA graph 关闭**：vLLM 需 `enforce_eager=True`；XPU 上的 graph 捕获尚未验证。
- **多模态（Qwen-VL）阻塞**：ViT SDPA 算子触发硬件级 `kl3ChannelCheckErrors` / `noc idle timeout`，需物理重置卡。请使用纯文本模型。
- **仅支持单机**：`XCCLCheckpointEngine` 的多机（节点间/节点内）广播已实现但尚未验证，当前仅验证单机 relay 模式。
- **vllm_kunlun 0.25.1 的 `logprobs`**：logprobs 路径可能返回 NaN/未初始化的 top-logprob token id，导致 tokenizer `OverflowError`。已反馈厂商。

## 平台内部实现（参考）

XPU 适配集中在少数文件：

- `src/twinkle/utils/platforms/xpu.py` — `XPU` 平台（继承 `GPU`）；`ensure_xpu_compat()` 桩化 `torch.xpu` 中的原生 Intel-XPU 桩；为 vLLM 提供设备 UUID 降级链（`current_platform` → `xpu-smi -q` Bus Id → `xpu-smi -L` UUID → sha1）。
- `src/twinkle/utils/platforms/base.py` — 通过 `PATH` 中的 `xpu-smi` 检测 XPU。
- `src/twinkle/infra/_ray/ray_helper.py` — 经 `ray.init(num_gpus=N)` 注册 cuda-alike 的 GPU 数量（Ray 无法自动发现 XPU）。经 `ray start` 启动的远程 worker 需手动传 `--num-gpus`。
- `src/twinkle/checkpoint_engine/xpu_checkpoint_engine.py` — `XCCLCheckpointEngine`（详见检查点引擎文档）。

## 参考资源

- [vLLM-Kunlun](https://github.com/baidu/vLLM-Kunlun)
- [Twinkle GitHub](https://github.com/modelscope/twinkle)
- [Twinkle 文档](https://twinkle.readthedocs.io/)

## 获取帮助

如果您在使用过程中遇到问题：

1. **提交 Issue**：[Twinkle GitHub Issues](https://github.com/modelscope/twinkle/issues)
2. **厂商栈问题**（LoRA 算子、logprobs、多模态）：请反馈昆仑芯 / vLLM-Kunlun。
