# CosineWarmupScheduler

这个 LRScheduler 用于在训练初始对学习率进行 warmup，在到达指定学习率后对学习率进行衰减。

```python
class CosineWarmupScheduler:

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5):
        ...

    ...
```

构造参数：
- optimizer: optimizer 优化器实例
- num_warmup_steps: warmup 的步数
- num_training_steps: 总训练的步数
- num_cycles: cosine 曲线周期，默认 0.5 半个余弦周期，即从最大学习率衰减到最小。调节为 1 为从最大学习率衰减到最小再回到最大。

这些参数可以通过模型的 `set_lr_scheduler` 来设置：

```python
model.set_lr_scheduler(CosineWarmupScheduler, num_warmup_steps=10, num_training_steps=100, num_cycles=0.5)
```

optimizer 参数不需要传入，模型模块内部会自动添加。

> Megatron 模型不支持该 Scheduler。