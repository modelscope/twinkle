# TrainMetric

Training metrics are used to measure the state during the training process. Training metrics include current learning rate, current step, total training time, training speed and other training metrics.

```python
from twinkle.metric import TrainMetric
metric = TrainMetric()
metric.accumulate(None, None, lr=0.0001, step=10, gradient_accumulation_steps=16)
...
_metric = metric.calculate()
```

> TrainMetric does not need device_mesh and process_group information, nor does it need inputs and outputs information
