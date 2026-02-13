# Twinkle Kernel 模块

Twinkle Kernel 模块提供了两条内核替换路径，用于加速训练和推理：

* **层级 Kernelize（Layer-level kernelize）**
  使用优化内核替换完整的 `nn.Module` 实现。
* **函数级 Kernelize（Function-level kernelize）**
  对 Python 模块中的特定函数进行 monkey-patch。

这两种方式可以独立使用，也可以通过统一入口组合使用。

---

## 概览：两条 Kernelize 路径

| 路径 | 粒度 | 典型场景 |
| --- | --- | --- |
| 层级替换 | 整个 `nn.Module` | Linear / Conv / MLP / Attention |
| 函数级替换 | 单个函数 | 热点路径、数学算子、激活函数 |

---

## 层级内核替换（Layer-Level）

### 适用场景

* 你已经有完整的层内核实现
* 希望在模型中批量替换某类 `nn.Module`
* 同时适用于训练与推理

---

### 示例 1：本地 Kernel 仓库

适用于：

* 内核实现位于本地仓库
* 希望替换 HuggingFace 或自定义模型中的层

```python
from twinkle.kernel import (
    kernelize_model,
    register_layer_kernel,
    register_external_layer,
)
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

# 1) 从本地仓库注册层内核
register_layer_kernel(
    kernel_name="MyAwesomeMLP",
    repo_path="/path/to/local/repo",
    package_name="my_kernels",
    layer_name="Qwen2MLPTrainingKernel",
    device="cuda",
    mode="train",
)

# 2) 绑定外部层与内核名
register_external_layer(Qwen2MLP, "MyAwesomeMLP")

# 3) 构建模型并应用内核替换
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

### 示例 2：Hub Kernel 仓库

适用于：

* 内核托管在 Hub 上

```python
import torch
import torch.nn as nn
from twinkle.kernel import (
    kernelize_model,
    register_layer_kernel,
    register_external_layer,
)

# 1) 定义自定义层
class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return nn.functional.silu(x1) * x2

# 2) 注册 Hub 内核并绑定层
register_layer_kernel(
    kernel_name="SiluAndMulKernel",
    repo_id="kernels-community/activation",
    layer_name="SiluAndMul",
    device="cuda",
    mode="train",
)
register_external_layer(SiluAndMul, "SiluAndMulKernel")

# 3) 应用到模型
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

## 本地 Kernel 仓库（最小结构）

本地 kernel 仓库本质上是一个普通 Python 包。
最少只需要一个 `layers.py` 来放层级内核实现。

```text
# 仓库结构：
my_kernels/                  # 本地 kernel 仓库（Python 包）
├── __init__.py              # 包入口
└── layers.py                # 层级 kernel 实现
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

## 函数级内核替换（Function-Level）

### 适用场景

* 只需要加速少量热点函数
* 不适合或不需要替换整个层
* 常用于数学算子、激活函数、工具函数

---

### 示例 1：批量注册（简单场景）

```python
from twinkle.kernel import register_kernels, kernelize_model

# 1) 注册函数内核
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

# 2) 应用（仅函数替换时 model 可为 None）
kernelize_model(model=None, mode="inference", device="cuda", use_fallback=True)
```

---

### 示例 2：高级函数来源（完整控制）

适用于：

* 不同函数来自不同来源（impl / repo / hub），或需要 compile/backward 等标志。

```python
from twinkle.kernel.function import (
    register_function_kernel,
    apply_function_kernel,
)
import torch.nn as nn
from twinkle.kernel import kernelize_model

TARGET_MODULE = "my_pkg.math_ops"

# 1) 直接传入实现
def fast_add(x, y):
    return x + y + 1

register_function_kernel(
    func_name="add",
    target_module=TARGET_MODULE,
    func_impl=fast_add,
    device="cuda",
    mode="inference",
)

# 2) Repo 对象（FuncRepositoryProtocol）
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

# 3) Hub 仓库
register_function_kernel(
    func_name="silu_and_mul",
    target_module="my_pkg.activations",
    repo_id="kernels-community/activation",
    revision="main",  # 或 version="0.1.0"
    device="cuda",
    mode="inference",
)

# 4) 应用函数内核
applied = apply_function_kernel(
    target_module=TARGET_MODULE,
    device="cuda",
    mode="inference",
    strict=False,
)
print("patched:", applied)

# 5) 可选：通过 kernelize_model 统一应用
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
kernelize_model(model=model, mode="inference", device="cuda", use_fallback=True)
```

---

## 层级 + 函数级统一批量注册

### 适用场景

* 需要框架级统一集成
* 希望通过单一配置入口管理
* 同时管理层和函数两类内核

```python
from twinkle.kernel import register_kernels, kernelize_model
import torch.nn as nn

# 1) 注册层级 + 函数级内核
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

# 2) 通过 kernelize_model 应用
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
kernelize_model(model=model, mode="train", device="cuda", use_fallback=True)
```
