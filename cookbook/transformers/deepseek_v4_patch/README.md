# DeepSeek-V4 NPU Sparse Attention (SAS) / Lightning Indexer (LI)

Twinkle 提供的 DeepSeek-V4 NPU 加速 patch，通过 monkey-patch 方式替换 transformers 中的注意力计算和索引器实现，无需修改 transformers 源码。

## 功能说明

### SAS (Sparse Attention Shared-KV)

替换 `DeepseekV4Attention.forward` 中的标准注意力计算，使用 mindspeed 提供的融合稀疏注意力核 `SparseAttnSharedKV`，支持三种注意力层类型：

- **Sliding Attention**: 纯滑动窗口注意力
- **CSA (Compressed Sparse Attention)**: 压缩稀疏注意力，使用 Lightning Indexer 选择 top-k 压缩条目
- **HCA (Heavily Compressed Attention)**: 高度压缩注意力，所有压缩条目可见

### LI (Lightning Indexer)

替换 `DeepseekV4Indexer.forward` 中的 torch 实现，使用 mindspeed 提供的 `npu_lightning_indexer` 加速 top-k 索引选择。

**注意**: 当前版本 SAS 和 LI 不能同时启用。

## 依赖

- **[ops-transformer](https://gitcode.com/cann/ops-transformer)**: 提供 NPU 算子实现，需要编译安装
- **[mindspeed](https://gitcode.com/Ascend/MindSpeed)**: 提供 NPU 算子调用实现，需要使用git clone下载mindspeed并切换到master分支进行手动安装
  - `mindspeed.ops.npu_sparse_attn_shared_kv.SparseAttnSharedKV` (SAS)
  - `mindspeed.ops.npu_lightning_indexer` (LI)
- **transformers**: 需包含 DeepSeek-V4 模型支持
- **torch_npu**: Ascend NPU 运行时

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TWINKLE_NPU_DSV4_SAS` | `0` | 启用 SAS patch |
| `TWINKLE_NPU_DSV4_LI` | `0` | 启用 LI patch |

**约束**: `TWINKLE_NPU_DSV4_SAS` 和 `TWINKLE_NPU_DSV4_LI` 不能同时设置为 `1`。

## 使用示例

### 镜像（可选）
```shell
#A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/ascend-sact/twinkle-npu:v4
```

### 启用 SAS

```bash
export TWINKLE_NPU_DSV4_SAS=1
torchrun --standalone --nnodes=1 --nproc_per_node=8 ep_fsdp2_lora_deepseek_v4_npu.py
```

### 启用 LI

```bash
export TWINKLE_NPU_DSV4_LI=1

torchrun --standalone --nnodes=1 --nproc_per_node=8 ep_fsdp2_lora_deepseek_v4_npu.py
```

### 完整示例脚本 (ds16_sas.sh)

```bash
#!/bin/bash
export GLOO_SOCKET_IFNAME="enp162s0f0"
export HCCL_SOCKET_IFNAME="enp162s0f0"
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=30000
export BATCH_SIZE=8
export MAX_STEPS=10
export GRADIENT_CHECKPOINTING=1
export USE_EP=1

# 启用 twinkle SAS patch
export TWINKLE_NPU_DSV4_SAS=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 ep_fsdp2_lora_deepseek_v4_npu.py
```

## 实现原理

Patch 在 `apply_npu_patch()` 阶段自动应用（位于 EP sharding 之后、FSDP wrap 之前），通过以下方式替换原始实现：

1. **Compressor patch**: 包装 `DeepseekV4HCACompressor` 和 `DeepseekV4CSACompressor` 的 `forward` 方法，确保返回 3-tuple `(compressed_kv, block_bias, top_k_indices)`
2. **Attention patch**: 替换 `DeepseekV4Attention.forward`，调用 `SparseAttnSharedKV.apply()` 替代标准注意力 dispatch
3. **Indexer patch**: 替换 `DeepseekV4Indexer.forward`，调用 `mindspeed.ops.npu_lightning_indexer` 替代 torch 实现

所有 patch 均包含 `ImportError` fallback，当 mindspeed 不可用时自动回退到原始实现。

## 验证

运行测试后，检查日志中是否出现：

```
[NPU] [DSV4-SAS] Twinkle sparse attention active (layer_type=..., cmp_ratio=..., topk=...)
```

或

```
[NPU] [DSV4-LI] Twinkle lightning indexer active (sparse_count=..., cmp_ratio=...)
```

## 相关文件

- `src/twinkle/kernel/deepseek_v4_npu.py`: Patch 核心实现
- `src/twinkle/kernel/monkey_patch_npu.py`: Patch 注册和环境变量控制
