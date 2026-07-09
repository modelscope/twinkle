# Padding-Free 训练

Padding-free（也称为"打包"）训练通过将多个序列拼接到一个打包批次中，消除了对 padding token 的无效计算。Twinkle 支持标准注意力和 Qwen3.5 GatedDeltaNet 线性注意力的 padding-free 训练。

## 工作原理

不同于将所有序列填充到 `max_length`，padding-free 将多个序列打包到一行中，并使用 `position_ids` 跟踪序列边界，从而避免在 padding token 上浪费算力。

```
标准方式:   [tok tok tok PAD PAD PAD]  [tok tok PAD PAD PAD PAD]
打包方式:   [tok tok tok tok tok ...]   ← 无 padding 浪费
```

## 使用方式

通过 `PackingDataset` 或 `IterablePackingDataset` 启用：

```python
from twinkle.dataset import PackingDataset

dataset = PackingDataset(
    dataset=base_dataset,
    max_length=8192,
)
```

数据集会自动打包序列并生成正确的 `position_ids`，在序列边界处重置。

## GatedDeltaNet 补丁（Qwen3.5）

Qwen3.5 使用混合架构，融合了标准注意力和 GatedDeltaNet 线性注意力。原生 GatedDeltaNet 实现不会在打包序列边界处重置线性注意力状态。

`GatedDeltaNetPaddingFreePatch` 通过以下方式修复：

1. Patch `Qwen3_5DecoderLayer.forward`，将 `cu_seq_lens_q`（累积序列长度）传递给线性注意力层
2. Patch `Qwen3_5GatedDeltaNet.forward`，使用支持 `cu_seqlens` 的 flash-linear-attention 内核（`causal_conv1d`、`chunk_gated_delta_rule`）

在 Qwen3.5 模型上检测到 padding-free 时，补丁会自动应用。

### 要求

- 需安装 `flash-linear-attention` 包
- 仅适用于含 GatedDeltaNet 层的 Qwen3.5 模型
- 启用序列并行时，会使用 `Qwen3_5GatedDeltaNetUlyssesPatch` 替代

## 注意力后端要求

| 注意力后端 | Padding-Free 支持 |
|-----------|-------------------|
| FlashAttention2 | 完全支持 |
| SDPA | 支持（不兼容序列并行） |
| Eager | 不支持 |
