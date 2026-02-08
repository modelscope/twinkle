# Accuracy

The accuracy metric is used to measure token-level accuracy information during training.

```python
from twinkle.metric import Accuracy
from twinkle.data_format import InputFeature, ModelOutput
metric = Accuracy(device_mesh=..., process_group=...)
metric.accumulate(InputFeature(labels=...), ModelOutput(logits=...))
...
_metric = metric.calculate()
```

> Accuracy does not currently support List[InputFeature] as input, meaning support for Megatron is yet to be adapted.
