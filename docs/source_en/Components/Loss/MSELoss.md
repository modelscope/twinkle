# MSE Loss

Mean Squared Error loss for regression-style training tasks.

```python
from twinkle.loss import MSELoss

loss_fn = MSELoss()
model.set_loss(loss_fn)
```

MSELoss computes the mean squared error between model output logits and the target labels. It is useful for tasks such as reward model training or value function estimation.
