# LinearWarmupScheduler

这个 LRScheduler 用于在训练初始对学习率进行 warmup，在到达指定学习率后对学习率进行衰减。

```python
class LinearWarmupScheduler:

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int):
        ...

    ...
```

构造参数：
- optimizer: optimizer 优化器实例
- num_warmup_steps: warmup 的步数
- num_training_steps: 总训练的步数

这些参数可以通过模型的 `set_lr_scheduler` 来设置：

```python
model.set_lr_scheduler(LinearWarmupScheduler, num_warmup_steps=10, num_training_steps=100)
```

optimizer 参数不需要传入，模型模块内部会自动添加。

> Megatron 模型不支持该 Scheduler。