# Patch

Patch is used to patch models. Patch is not needed in most cases, but it may be needed when changing training tasks or when the model's own code has bugs.

For example:
```python
model.apply_patch('ms://twinkle-kit/qwen3_moe_transformers4_patch')
```

You can also:
```python
from twinkle.patch import apply_patch
apply_patch(module, 'ms://twinkle-kit/qwen3_moe_transformers4_patch')
```
This method is suitable if you use other frameworks for training or inference, but use twinkle-kit's patch for patching.

The base class of Patch is relatively simple:
```python
class Patch:

    def patch(self, module, *args, **kwargs) -> None:
        ...
```

> Patch is strongly recommended to be placed in the ModelScope or Hugging Face model repository and loaded remotely. Because there may be many Patches and they are fragmented.
