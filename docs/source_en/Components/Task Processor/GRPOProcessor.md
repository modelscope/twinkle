# GRPOLossProcessor

GRPOLossProcessor is a task processor wrapper designed for GRPO reinforcement learning training. It extends InputProcessor with GRPO-specific data preparation.

```python
from twinkle.processor import GRPOLossProcessor

processor = GRPOLossProcessor(
    device_mesh=...,
    padding_free=False,
    framework='transformers',
)

model.set_processor(processor)
```

GRPOLossProcessor wraps the base `InputProcessor` and adds handling for GRPO-specific fields such as advantages, old log-probabilities, and reference log-probabilities that are required by the GRPO loss function.

> For standard SFT tasks, use `InputProcessor` directly. Use `GRPOLossProcessor` when your training loop involves GRPO or its variants.
