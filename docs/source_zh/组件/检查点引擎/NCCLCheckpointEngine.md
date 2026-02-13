# NCCLCheckpointEngine

使用 NCCL 进行 GPU 间高速权重传输的检查点引擎。

## 使用示例

```python
from twinkle.checkpoint_engine import NCCLCheckpointEngine

# 在训练进程 (rank 0)
engine = NCCLCheckpointEngine(bucket_size=512<<20)  # 512MB bucket
engine.is_master = True
engine.prepare()
engine.init_process_group(rank=0, world_size=5)

# 发送权重
await engine.send_weights(model.named_parameters())
engine.finalize()

# 在推理进程 (rank 1-4)
engine = NCCLCheckpointEngine(bucket_size=512<<20)
engine.prepare()
engine.init_process_group(rank=1, world_size=5, master_metadata=metadata)

# 接收权重
async for name, tensor in engine.receive_weights():
    model.load_state_dict({name: tensor}, strict=False)
engine.finalize()
```

## 特性

- **高速传输**: 使用 NCCL 实现 GPU 间点对点高速传输
- **零拷贝**: 直接在 GPU 内存间传输,无需经过 CPU
- **分桶传输**: 支持大模型的分桶传输

## 配置参数

- **bucket_size**: 权重桶大小,控制每次传输的数据量。较大的桶可以提高传输效率,但会占用更多内存
- **timeout**: 传输超时时间

> NCCLCheckpointEngine 是 GPU 训练的推荐选择,提供最高的传输性能。
