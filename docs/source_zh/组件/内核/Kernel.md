# Twinkle Kernel 模块

`twinkle.kernel` 提供一个 mapping 驱动的内核替换接口，把“用一种实现替换模型里的另一种实现”压缩为一次 `kernelize(model, mapping)` 调用。

公开符号只有三个：

| 符号 | 作用 |
| --- | --- |
| `kernelize(model, mapping)` | 在 `model` 上应用 `mapping`，原地修改后返回 |
| `npu_builtin(model=None)` | 返回 Ascend NPU 内置替换的 mapping dict（可与用户 mapping 自由组合） |
| `hub(ref, *, revision=None, version=None, backend=None, trust_remote_code=False)` | 构造一个 `HubRef`，用作 mapping value；真实下载推迟到 `kernelize` 执行 |

## Mapping 语义

`mapping` 的 **key** 表示要替换的目标：

- `type[nn.Module]` 子类：替换模型里**所有**该精确类型的实例（`m.__class__ = impl_class`，**不包含**子类）
- `str` 形如 `'pkg.sub.attr'` 或 `'pkg.sub.ClassName.attr'`：`setattr(target, attr, impl)`

**value** 表示用什么替换：

- `type[nn.Module]` 子类：直接作为 impl 类。该类**不会被 `__init__` 调用**，必须只依赖原 instance 已经有的 attribute（weight / eps / ...）正确工作
- `Callable`：直接 `setattr` 上去
- `dict[str, V]`：device → impl 嵌套分派。从 `model` 推断当前 device，未匹配则**静默跳过**
- `HubRef`：通过 `hub(...)` 构造的 Hub 引用，延迟加载

device 从 `next(model.parameters()).device.type` 推断（无参数则用 buffers，再无则为 `'cpu'`）。

## 场景示例

### 启用全部 NPU 内置优化

```python
import torch
from twinkle.kernel import kernelize, npu_builtin

if torch.npu.is_available():
    model = kernelize(model, npu_builtin(model))
```

### 自定义类替换

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from twinkle.kernel import kernelize

model = kernelize(model, {Qwen2RMSNorm: MyRMSNorm})
```

### 内置 + 自定义混合

```python
from twinkle.kernel import kernelize, npu_builtin

model = kernelize(model, {**npu_builtin(model), Qwen2RMSNorm: MyRMSNorm})
```

后写入的 key 会覆盖前面的，普通 dict 合并语义。

### Hub Kernel（HF Hub 格式）

```python
from twinkle.kernel import kernelize, hub
from my_pkg import SiluAndMul

model = kernelize(model, {
    SiluAndMul: hub('kernels-community/activation:SiluAndMul', version=1),
})
```

`revision` 与 `version` 二选一必传。`hub(...)` 触发 `kernels` 包的延迟 import，未安装时会提示 `pip install kernels`。

### 函数级替换

```python
from twinkle.kernel import kernelize
from twinkle.kernel.npu_impls.rotary import npu_apply_rotary_pos_emb

model = kernelize(model, {
    'transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb':
        npu_apply_rotary_pos_emb,
})
```

### 跨设备 mapping（NPU 启用、CUDA 跳过）

```python
from twinkle.kernel import kernelize

model = kernelize(model, {
    Qwen2RMSNorm: {'npu': NpuRMSNorm, 'cuda': CudaRMSNorm},
})
```

在 CUDA 模型上跑也安全：未匹配 device 的 entry 不会替换、不会报错。

## 内置 NPU 优化

`npu_builtin(model)` 返回的 dict 至少包含以下覆盖（实际条目随 transformers 已安装的 modeling 模块动态收集）：

- Qwen2 / Qwen3 / Qwen3-MoE / Qwen2.5-VL / Qwen3.5 / Qwen3.5-MoE 系列的 RMSNorm 类替换
- 同上系列的 `apply_rotary_pos_emb` 函数替换（融合 RoPE）
- 同上系列 MLP 的 SwiGLU 融合替换
- Qwen3-MoE / Qwen3.5-MoE 的 `Experts.forward` 与 `SparseMoeBlock.forward` 替换
- Qwen3.5 / Qwen3.5-MoE 的 GatedRMSNorm forward 替换
- Qwen2.5-VL 的 `apply_multimodal_rotary_pos_emb` 替换
- 全局 SDPA 替换（一次性副作用，写入 `ALL_ATTENTION_FUNCTIONS['sdpa']`）
- Qwen3.5 Flash Linear Attention 启用（一次性副作用 + 实例遍历，由 `npu_builtin(model)` 内部触发）

**未默认包含** `transformers.integrations.moe._grouped_mm` 的 NPU 替换（在没有 Expert Parallelism 时会带来约 8x 开销）。需要时手动加入：

```python
from twinkle.kernel import kernelize, npu_builtin
from twinkle.kernel.npu_impls.moe import npu_grouped_mm

mapping = {
    **npu_builtin(model),
    'transformers.integrations.moe._grouped_mm': {'npu': npu_grouped_mm},
}
model = kernelize(model, mapping)
```

## 环境变量

只有两个保留：

- `TWINKLE_NPU_FLA`：Qwen3.5 FLA 开关（默认开，设为 `0`/`false` 关闭）
- `TWINKLE_NPU_GATED_RMSNorm_FP32`：将 Gated RMSNorm 强制升到 FP32 计算（默认关）

旧的 `TWINKLE_NPU_PATCH` / `TWINKLE_NPU_FUSED_OPS` / `TWINKLE_NPU_GMM_PATCH` / `TWINKLE_USE_KERNELS` 已移除——这些都改成"是否把对应 entry 写进 mapping"的显式选择。

## 注意事项

- `m.__class__ = impl_cls` 是 Python class 替换魔法。impl 类**必须**只覆盖 `forward`（以及辅助方法），不能定义 `__init__`，否则原 instance 的 attribute 会与 impl 的预期错位
- 精确匹配：`type(m) is target_cls`。继承自 `target_cls` 的子类不会被替换；如需替换，把子类也放进 mapping
- 调用 `kernelize` 多次是幂等的（`__class__` 已是 impl 时再设一次无害）
- 没有 `unkernelize`——替换是单向的
