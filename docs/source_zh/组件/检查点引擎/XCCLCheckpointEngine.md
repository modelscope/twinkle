# XCCLCheckpointEngine

面向昆仑芯 XPU 的检查点引擎，经 BKCL（以 `nccl`/XCCL 后端挂载到 PyTorch）进行权重传输。

## 使用示例

```python
from twinkle.checkpoint_engine import XCCLCheckpointEngine

engine = XCCLCheckpointEngine(bucket_size=512<<20)
# 使用方式与 NCCLCheckpointEngine 相同
```

在 XPU 上该引擎由 `CheckpointEngineManager` / `CheckpointEngineMixin` 自动选择，通常无需手动构造。

## 特性

- **NCCL 直替**：继承 `NCCLCheckpointEngine`，分桶、ZMQ 元数据握手、双缓冲等逻辑原样复用。
- **XCCL 组 + relay 兜底**：将直接构造 `ProcessGroupNCCL`（当两个 rank 共享同一 local device index 时会死锁）替换为 stateless 的 `ProcessGroupXCCL`，并对冲突 rank 提供 relay 路径。
- **基于 store 的初始化 barrier**：就绪 barrier 走 TCPStore，而非设备集合通信。

## 为何需要独立引擎

BKCL 按 **local device index** 识别 rank 设备，且不做 `CUDA_VISIBLE_DEVICES` → 物理卡的翻译。在单机上分离部署 trainer/sampler 时，两侧可见设备均为 local `0`，产生重复的 `(host, index)` 对。此时直接构造 `ProcessGroupNCCL` 无法成环，首次 broadcast 死锁（600s WorkXCCL 超时）。

`XCCLCheckpointEngine` 经 store 交换 `(hostname, local_device_index)`，将冲突 rank 排除出 XCCL 组，并由最小 rank 的 XCCL 成员经直连 TCP socket 中继（relay）。无冲突时使用平铺的 stateless `ProcessGroupXCCL`。

## 适用场景

- 昆仑芯 XPU 上的权重同步（训练 ↔ 采样）
- colocate / 分离式 RL（GRPO）下的全量权重同步（`merge_and_sync=True`）

## 限制

- 多机（节点间/节点内）广播已实现但尚未验证，当前仅验证单机 relay 模式。

> 在昆仑芯 XPU 上权重同步请使用 `merge_and_sync=True`；LoRA 采样当前受厂商算子 dtype 限制而阻塞（详见 [XPU 的支持](../../使用指引/XPU的支持.md) 指南）。
