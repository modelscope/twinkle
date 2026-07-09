# GRPOMetric

`GRPOMetric` 跟踪 GRPO 训练中的策略优化诊断指标，包括 KL 散度、裁剪率、熵和对数概率统计。

## 使用方法

```python
from twinkle.metric import GRPOMetric

metric = GRPOMetric(
    device_mesh=device_mesh,
    process_group=process_group,
    epsilon=0.2,          # PPO 裁剪范围
    temperature=1.0,      # 用于 logp 重缩放的采样温度
    top_k_kl=10,          # 每步记录 top-K 高 KL token
)

# 训练循环中
metric.accumulate(inputs, outputs, old_logps=old_logps, advantages=advantages)

# 日志间隔时
results = metric.calculate()
```

## 输出指标

| 指标 | 说明 |
|:-----|:-----|
| `train/policy_confidence` | exp(mean_new_logp) — 越高表示模型越自信 |
| `train/mean_new_logp` | 当前策略下生成 token 的平均对数概率 |
| `train/mean_old_logp` | 参考策略下的平均对数概率 |
| `train/approx_kl` | Schulman K3 KL 估计器 |
| `train/entropy` | 平均 token 级熵 |
| `train/clip_ratio` | 被裁剪的 token 比例 |

## 变体

- **`GSPOMetric`** — 序列级裁剪率（几何平均比率）
- **`CISPOMetric`** — 无条件裁剪率（不按优势符号门控）
