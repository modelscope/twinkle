# LossMetric

LossMetric用于打印和评估残差和grad_norm信息

```python
from twinkle.metric import LossMetric
from twinkle.data_format import InputFeature, ModelOutput
metric = LossMetric(device_mesh=..., process_group=...)
metric.accumulate(InputFeature(labels=...), ModelOutput(loss=...), grad_norm=...)
...
_metric = metric.calculate()
```
