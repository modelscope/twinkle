# EmbeddingMetric

`EmbeddingMetric` 跟踪对比学习（InfoNCE）训练中的嵌入质量，报告锚点-正样本余弦相似度和批内负样本相似度。

## 使用方法

```python
from twinkle.metric import EmbeddingMetric

metric = EmbeddingMetric(device_mesh=device_mesh, process_group=process_group)

# 训练中
metric.accumulate(inputs, outputs)

# 日志间隔时
results = metric.calculate()
# results: {'pos_sim': '0.8523', 'neg_sim': '0.2134', 'loss': '0.3412', ...}
```

## 输出指标

| 指标 | 说明 |
|:-----|:-----|
| `pos_sim` | 锚点与正样本的平均余弦相似度 |
| `pos_sim_min` | 批内最小正样本相似度 |
| `pos_sim_max` | 批内最大正样本相似度 |
| `neg_sim` | 锚点与其他正样本（批内负样本）的平均相似度 |
| `loss` | 平均对比损失值 |
| `grad_norm` | 梯度范数 |

> 此指标与 `InfonceLoss` 配合使用，适用于嵌入/检索模型训练。
