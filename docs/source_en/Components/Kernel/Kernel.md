# Twinkle Kernel Module

The Twinkle Kernel Module provides two kernel replacement paths for accelerating models during training and inference:

* **Layer-level kernelize**
  Replace entire `nn.Module` implementations with optimized kernels.
* **Function-level kernelize**
  Monkey-patch specific functions inside a Python module.

These two approaches can be used independently or together via a unified registration and application entry point.

---

## Overview: Two Kernelization Paths

| Path           | Granularity          | Typical Use Cases                |
| -------------- | -------------------- | -------------------------------- |
| Layer-level    | Whole `nn.Module`    | Linear / Conv / MLP / Attention  |
| Function-level | Individual functions | Hot paths, math ops, activations |

---

## Layer-Level Kernel Replacement

### When to Use

* You have a complete kernel implementation for a layer
* You want model-wide replacement of specific `nn.Module` types
* Suitable for both training and inference

---

### Example 1: Local Kernel Repo

Use this when:

* Kernel implementations live in a local repository
* You want to replace layers in HuggingFace or custom models

```python
from twinkle.kernel import (
    kernelize_model,
    register_layer_kernel,
    register_external_layer,
)
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

# 1) Register the layer kernel from a local repo
register_layer_kernel(
    kernel_name="MyAwesomeMLP",
    repo_path="/path/to/local/repo",
    package_name="my_kernels",
    layer_name="Qwen2MLPTrainingKernel",
    device="cuda",
    mode="train",
)

# 2) Bind external layer to kernel name
register_external_layer(Qwen2MLP, "MyAwesomeMLP")

# 3) Build the model and apply kernelization
config = Qwen2Config(
    hidden_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=256,
    use_cache=False,
)
model = Qwen2ForCausalLM(config)
model = kernelize_model(model, mode="train", device="cuda", use_fallback=True)
```

---

### Example 2: Hub Kernel Repo

Use this when:

* The kernel is hosted on a Hub

```python
import torch
import torch.nn as nn
from twinkle.kernel import (
    kernelize_model,
    register_layer_kernel,
    register_external_layer,
)

# 1) Define the custom layer
class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return nn.functional.silu(x1) * x2

# 2) Register the Hub kernel and bind the layer
register_layer_kernel(
    kernel_name="SiluAndMulKernel",
    repo_id="kernels-community/activation",
    layer_name="SiluAndMul",
    device="cuda",
    mode="train",
)
register_external_layer(SiluAndMul, "SiluAndMulKernel")

# 3) Apply to a model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)

model = SimpleModel()
model = kernelize_model(model, mode="train", device="cuda", use_fallback=True)
```

---

## Local Kernel Repo (Minimal)

A local kernel repository is a regular Python package.
At minimum, it only needs a `layers.py` file for layer-level kernels.

```text
# Repo layout:
my_kernels/                  # Local kernel repository (Python package)
├── __init__.py              # Package entry
└── layers.py                # Layer-level kernel implementations

```

```python
# my_kernels/__init__.py
from . import layers
__all__ = ["layers"]

# my_kernels/layers.py
import torch
import torch.nn as nn

class Qwen2MLPTrainingKernel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(gate) * up)
```

---

## Function-Level Kernel Replacement

### When to Use

* You only need to accelerate a small number of hot functions
* Replacing the entire layer is unnecessary or impractical
* Common for math ops, activations, or utility functions

---

### Example 1: Batch Registration (Simple Case)

```python
from twinkle.kernel import register_kernels, kernelize_model

# 1) Register function kernels
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

# 2) Apply (model can be None when only functions are used)
kernelize_model(model=None, mode="inference", device="cuda", use_fallback=True)
```

---

### Example 2: Advanced Function Sources (Full Control)

Use this when:

* Use when different functions come from different sources (impl / repo / hub) or need compile/backward flags.

```python
from twinkle.kernel.function import (
    register_function_kernel,
    apply_function_kernel,
)
import torch.nn as nn
from twinkle.kernel import kernelize_model

TARGET_MODULE = "my_pkg.math_ops"

# 1) Direct implementation
def fast_add(x, y):
    return x + y + 1

register_function_kernel(
    func_name="add",
    target_module=TARGET_MODULE,
    func_impl=fast_add,
    device="cuda",
    mode="inference",
)

# 2) Repo object (FuncRepositoryProtocol)
class MyFuncRepo:
    def load(self):
        return MyKernelFunc

class MyKernelFunc(nn.Module):
    def forward(self, x, y):
        return x * y

register_function_kernel(
    func_name="mul",
    target_module=TARGET_MODULE,
    repo=MyFuncRepo(),
    device="cuda",
    mode="compile",
)

# 3) Hub repo
register_function_kernel(
    func_name="silu_and_mul",
    target_module="my_pkg.activations",
    repo_id="kernels-community/activation",
    revision="main",  # or version="0.1.0"
    device="cuda",
    mode="inference",
)

# 4) Apply function kernels
applied = apply_function_kernel(
    target_module=TARGET_MODULE,
    device="cuda",
    mode="inference",
    strict=False,
)
print("patched:", applied)

# 5) Optional: unified entry via kernelize_model
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
kernelize_model(model=model, mode="inference", device="cuda", use_fallback=True)
```

---

## Unified Layer + Function Batch Registration

### When to Use

* Framework-level integration
* A single configuration entry point is preferred
* Managing both layer and function kernels together

```python
from twinkle.kernel import register_kernels, kernelize_model
import torch.nn as nn

# 1) Register layer + function kernels
config = {
    "layers": {
        "linear": {
            "repo_id": "kernels-community/linear",
            "layer_name": "Linear",
            "version": "0.1.0",
            "device": "cuda",
            "mode": "train",
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

# 2) Apply via kernelize_model
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
kernelize_model(model=model, mode="train", device="cuda", use_fallback=True)
```
