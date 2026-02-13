# Accuracy

准确率指标用于衡量训练时的token级别准确率信息。

```python
from twinkle.metric import Accuracy
from twinkle.data_format import InputFeature, ModelOutput
metric = Accuracy(device_mesh=..., process_group=...)
metric.accumulate(InputFeature(labels=...), ModelOutput(logits=...))
...
_metric = metric.calculate()
```

> Accuracy目前尚未支持List\[InputFeature\]作为输入，也就是对Megatron的支持待适配。
