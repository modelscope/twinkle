User scenarios and examples

1) You want to speed up whole model layers (layer-level kernelize).
Use when you have kernel repos for modules like Linear/Conv and want model-wide replacement.

```python
from twinkle.kernel import kernelize_model, register_layer_kernel
import torch.nn as nn

register_layer_kernel(
    kernel_name="linear",
    repo_id="kernels-community/linear",
    device="cuda",
    mode="inference",
)

model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
model = kernelize_model(model, mode="inference", device="cuda", use_fallback=True)
```

2) You want to speed up a few hot functions (function-level kernelize).
Use when you only need to patch specific functions in a module.

```python
from twinkle.kernel import kernelize_model, register_kernels

config = {
    "functions": {
        "add": {
            "target_module": "my_pkg.math_ops",
            "func_impl": lambda x, y: x + y + 1,
            "device": "cuda",
            "mode": "inference",
        },
    },
}
register_kernels(config)

# model can be None when you only need function kernels
kernelize_model(model=None, mode="inference", device="cuda", use_fallback=True)
```

3) You want full control over function kernel sources (impl / repo / hub).
Use when different functions come from different sources or need compile/backward flags.

```python
from twinkle.kernel.function import register_function_kernel, apply_function_kernel

TARGET_MODULE = "my_pkg.math_ops"

# 1) Direct implementation
def fast_add(x, y):
    return x + y + 1

register_function_kernel(
    func_name="add",
    target_module=TARGET_MODULE,
    func_impl=fast_add,
    device="cpu",
    mode="inference",
)

# 2) Repo object (FuncRepositoryProtocol)
class MyFuncRepo:
    def load(self):
        return MyKernelFunc

class MyKernelFunc:
    can_torch_compile = True
    has_backward = True
    def __call__(self, x, y):
        return x + y + 2

register_function_kernel(
    func_name="mul",
    target_module=TARGET_MODULE,
    repo=MyFuncRepo(),
    device="cpu",
    mode="compile",
)

# 3) Hub repo_id
register_function_kernel(
    func_name="silu_and_mul",
    target_module="my_pkg.activations",
    repo_id="kernels-community/activation",
    revision="main",  # or version="0.1.0"
    device="cpu",
    mode="inference",
)

applied = apply_function_kernel(
    target_module=TARGET_MODULE,
    device="cpu",
    mode="inference",
    use_fallback=True,
    strict=False,
)

print("patched:", applied)

# Optional: apply via kernelize_model as the entry point
from twinkle.kernel import kernelize_model
import torch.nn as nn

model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
kernelize_model(model=model, mode="inference", device="cpu", use_fallback=True)
```

4) You want one config to cover layers and functions.
Use when integrating into a framework or a single entry point is preferred.

```python
from twinkle.kernel import register_kernels

config = {
    "layers": {
        "linear": {
            "repo_id": "kernels-community/linear",
            "layer_name": "Linear",
            "version": "0.1.0",
            "device": "cuda",
            "mode": "inference",
        },
        "conv2d": {
            "repo_path": "/path/to/local/repo",
            "package_name": "my_kernels",
            "layer_name": "Conv2d",
            "device": "cuda",
        },
    },
    "functions": {
        "add": {
            "target_module": "my_pkg.math_ops",
            "func_impl": lambda x, y: x + y + 1,
            "device": "cuda",
            "mode": "inference",
        },
        "relu": {
            "target_module": "my_pkg.activations",
            "repo_id": "kernels-community/activation",
            "revision": "main",
            "device": "cuda",
        },
    },
}

register_kernels(config)

# Optional: apply both layer and function kernels via kernelize_model
from twinkle.kernel import kernelize_model
import torch.nn as nn

model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
kernelize_model(model=model, mode="inference", device="cuda", use_fallback=True)
```
