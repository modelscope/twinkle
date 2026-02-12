# LinearWarmupScheduler

This LRScheduler is used to warm up the learning rate at the beginning of training and decay the learning rate after reaching the specified learning rate.

```python
class LinearWarmupScheduler:

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int):
        ...

    ...
```

Construction parameters:
- optimizer: optimizer instance
- num_warmup_steps: Number of warmup steps
- num_training_steps: Total training steps

These parameters can be set through the model's `set_lr_scheduler`:

```python
model.set_lr_scheduler(LinearWarmupScheduler, num_warmup_steps=10, num_training_steps=100)
```

The optimizer parameter does not need to be passed in; the model module will automatically add it internally.

> Megatron models do not support this Scheduler.
