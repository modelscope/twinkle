# MSE 损失

均方误差损失，用于回归式训练任务。

```python
from twinkle.loss import MSELoss

loss_fn = MSELoss()
model.set_loss(loss_fn)
```

MSELoss 计算模型输出 logits 与目标 labels 之间的均方误差。适用于奖励模型训练或价值函数估计等任务。
