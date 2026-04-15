# Loss Scale

The LossScale component controls loss scaling for numerical stability during training, particularly useful for mixed-precision training.

```python
from twinkle.loss_scale import LossScale

loss_scale = LossScale()

# Apply scaling to loss before backward
scaled_loss = loss_scale(loss, num_tokens)
```

LossScale handles the normalization of loss values by the number of valid tokens, ensuring consistent gradient magnitudes across different batch sizes and sequence lengths.

> LossScale is used internally by the model's training pipeline. It is automatically applied when using `model.forward_backward()`.
