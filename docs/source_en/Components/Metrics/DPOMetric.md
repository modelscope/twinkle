# DPOMetric

The DPOMetric tracks preference optimization-specific statistics during DPO training.

```python
from twinkle.metric import DPOMetric

metric = DPOMetric(device_mesh=..., process_group=...)

# Accumulate after each forward pass
metric.accumulate(inputs, outputs, ref_outputs=ref_outputs)

# Calculate aggregated metrics
result = metric.calculate()
```

**Tracked metrics:**
- `chosen_logps`: Average log-probability of chosen responses
- `rejected_logps`: Average log-probability of rejected responses
- `ref_chosen_logps`: Reference model log-probability of chosen responses
- `ref_rejected_logps`: Reference model log-probability of rejected responses
- `rewards/chosen`: Implicit reward for chosen responses
- `rewards/rejected`: Implicit reward for rejected responses
- `accuracy`: Fraction of pairs where chosen is preferred over rejected
- `margin`: Average reward margin between chosen and rejected

> DPOMetric performs DP-aware aggregation across all data-parallel ranks. It supports both interleaved and separate chosen/rejected batch formats.
