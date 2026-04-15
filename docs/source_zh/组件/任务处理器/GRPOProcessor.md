# GRPOLossProcessor

GRPOLossProcessor 是专为 GRPO 强化学习训练设计的任务处理器包装器。它在 InputProcessor 基础上扩展了 GRPO 特有的数据准备功能。

```python
from twinkle.processor import GRPOLossProcessor

processor = GRPOLossProcessor(
    device_mesh=...,
    padding_free=False,
    framework='transformers',
)

model.set_processor(processor)
```

GRPOLossProcessor 包装了基础 `InputProcessor`，并添加了 GRPO 特有字段的处理，如优势值、旧对数概率和参考对数概率，这些是 GRPO 损失函数所需要的。

> 对于标准 SFT 任务，直接使用 `InputProcessor`。当训练循环涉及 GRPO 或其变体时，使用 `GRPOLossProcessor`。
