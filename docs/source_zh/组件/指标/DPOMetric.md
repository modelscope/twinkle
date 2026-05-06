# DPOMetric

DPOMetric 在 DPO 训练过程中跟踪偏好优化相关的统计数据。

```python
from twinkle.metric import DPOMetric

metric = DPOMetric(device_mesh=..., process_group=...)

# 每次前向传播后累积
metric.accumulate(inputs, outputs, ref_outputs=ref_outputs)

# 计算聚合指标
result = metric.calculate()
```

**跟踪的指标:**
- `chosen_logps`: chosen 响应的平均对数概率
- `rejected_logps`: rejected 响应的平均对数概率
- `ref_chosen_logps`: 参考模型对 chosen 响应的对数概率
- `ref_rejected_logps`: 参考模型对 rejected 响应的对数概率
- `rewards/chosen`: chosen 响应的隐式奖励
- `rewards/rejected`: rejected 响应的隐式奖励
- `accuracy`: chosen 优于 rejected 的样本对比例
- `margin`: chosen 和 rejected 之间的平均奖励差距

> DPOMetric 在所有数据并行 rank 上执行 DP 感知的聚合。支持交替排列和分开排列的 chosen/rejected 批次格式。
