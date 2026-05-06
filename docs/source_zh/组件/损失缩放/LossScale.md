# 损失缩放

LossScale 组件控制训练过程中的损失缩放，确保数值稳定性，在混合精度训练中尤为重要。

```python
from twinkle.loss_scale import LossScale

loss_scale = LossScale()

# 在反向传播前对损失进行缩放
scaled_loss = loss_scale(loss, num_tokens)
```

LossScale 通过有效 token 数量对损失值进行归一化，确保不同批次大小和序列长度下梯度幅度的一致性。

> LossScale 在模型训练流水线中内部使用。使用 `model.forward_backward()` 时会自动应用。
