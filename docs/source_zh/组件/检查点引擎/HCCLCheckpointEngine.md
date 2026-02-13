# HCCLCheckpointEngine

使用 HCCL 进行昇腾 NPU 间权重传输的检查点引擎。

## 使用示例

```python
from twinkle.checkpoint_engine import HCCLCheckpointEngine

engine = HCCLCheckpointEngine(bucket_size=512<<20)
# 使用方式与 NCCLCheckpointEngine 相同
```

## 特性

- **NPU 优化**: 专为昇腾 NPU 优化的权重传输
- **高效通信**: 使用 HCCL 实现 NPU 间高速通信
- **兼容接口**: 与 NCCLCheckpointEngine 保持一致的接口

## 适用场景

HCCLCheckpointEngine 专门用于昇腾 NPU 环境:

- 使用华为昇腾 NPU 进行训练
- 需要在 NPU 间同步模型权重
- 大规模 NPU 集群部署

> 在昇腾 NPU 环境中,HCCLCheckpointEngine 提供了与 NCCL 相当的性能。
