# LossMetric

LossMetric is used to print and evaluate loss and grad_norm information

```python
from twinkle.metric import LossMetric
from twinkle.data_format import InputFeature, ModelOutput
metric = LossMetric(device_mesh=..., process_group=...)
metric.accumulate(InputFeature(labels=...), ModelOutput(loss=...), grad_norm=...)
...
_metric = metric.calculate()
```
