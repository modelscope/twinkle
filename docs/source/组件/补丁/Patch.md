# Patch

Patch 用于对模型进行补丁。Patch 在大部分情况并不需要，但是在改变训练任务、模型本身代码存在 bug 的情况下是可能有需要的。

例如：
```python
model.apply_patch('ms://twinkle-kit/qwen3_moe_transformers4_patch')
```

也可以：
```python
from twinkle.patch import apply_patch
apply_patch(module, 'ms://twinkle-kit/qwen3_moe_transformers4_patch')
```
这种方式可以适合于你使用其他框架训练或推理，但使用 twinkle-kit 的 patch 打补丁的情况。

Patch 的基类比较简单：
```python
class Patch:

    def patch(self, module, *args, **kwargs) -> None:
        ...
```

> Patch 强烈建议放在 ModelScope 或者 Hugging Face 的模型库中，以远程方式加载。因为 Patch 可能数量较多而且细碎。