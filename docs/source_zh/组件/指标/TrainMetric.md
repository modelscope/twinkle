# TrainMetric

训练指标用于衡量训练过程中的状态。训练指标包含了当前学习率、当前step、总训练时长、训练速度等训练指标。

```python
from twinkle.metric import TrainMetric
metric = TrainMetric()
metric.accumulate(None, None, lr=0.0001, step=10, gradient_accumulation_steps=16)
...
_metric = metric.calculate()
```

> TrainMetric 不需要 device_mesh 和 process_group 信息，也不需要 inputs、outputs 信息