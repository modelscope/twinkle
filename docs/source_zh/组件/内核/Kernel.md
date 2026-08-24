# Twinkle Kernel 模块

`twinkle.kernel` 提供一个 mapping 驱动的内核替换接口，把"用一种实现替换模型里的另一种实现"压缩为一次 `kernelize(model, mapping)` 调用。三件事被彻底拆开：**替换什么**（mapping 的 key）、**选什么实现**（`KernelChoice`）、**怎么安装**（installer）。

公开符号只有四个：

| 符号 | 作用 |
| --- | --- |
| `kernelize(model, mapping=None)` | 在 `model` 上应用 `mapping`，原地修改后返回。省略 `mapping` 时应用内置的 `DEFAULT_KERNEL_CONFIG` |
| `KernelChoice(op, backends, installer=None)` | mapping value：本次替换用哪个 op、按什么优先级选 backend |
| `DEFAULT_KERNEL_CONFIG` | 内置默认 mapping（target → `KernelChoice`），复制/合并/覆盖即可自定义 |
| `hub(ref, *, revision=None, version=None, backend=None)` | 构造一个 `HubRef`，用作 mapping value；真实下载推迟到 `kernelize` 执行 |

## Mapping 语义

**key（target）两种形态**，与 value 自由组合：

- `type[nn.Module]` 子类：替换模型里**所有**该精确类型的实例（`m.__class__ = impl_class`，**不包含**子类）
- `str` 点路径：`'pkg.mod.ClassName'`（解析为 `nn.Module` 子类 → 等价类替换）、`'pkg.mod.ClassName.forward'`（类方法 → `setattr`）、`'pkg.mod.attr'`（模块函数 → `setattr`）。`transformers.*` 家族未安装时 DEBUG 静默跳过（家族缺失 = 常态）

**value 三种形式**：

- 直接 impl（类或函数）：不执行 backend 选择，直接用通用 installer 安装。impl 类**不会被 `__init__` 调用**，必须只依赖原 instance 已有的 attribute 正确工作
- `KernelChoice(op='rms_norm', backends=('npu', 'liger'))`：按 `backends` 顺序挑第一个可用实现——backend 未注册 / `available()` 不通过（含原因）/ 加载异常都自动顺延；全部不可用则保留原始实现
- `HubRef`：通过 `hub(...)` 构造的 Hub 引用，延迟加载

## 默认配置与自定义

```python
model = kernelize(model)
# 等价于
model = kernelize(model, DEFAULT_KERNEL_CONFIG)
```

`DEFAULT_KERNEL_CONFIG` 覆盖：Qwen2 / Qwen3 / Qwen3-MoE / Qwen2.5-VL / Qwen3.5 / Qwen3.5-MoE 的 RMSNorm / RoPE / SwiGLU / MoE（链 `('npu', 'liger')`，NPU 上 CANN 优先）；llama 系 8 族、gemma 系 4 族（链 `('liger',)`）；以及逻辑 target `'sdpa'`（全局 SDPA 安装）与 `'fla'`（Qwen3.5 Flash Linear Attention 逐实例 patch）。

**mapping 是全量替换默认配置而非增量**——只传自己那几条就意味着默认条目全部不应用。在默认基础上微调用复制合并：

```python
from twinkle.kernel import DEFAULT_KERNEL_CONFIG, KernelChoice, kernelize

model = kernelize(model, {**DEFAULT_KERNEL_CONFIG,
    # rms_norm 改成 liger 优先（liger 不可用自动退 npu）
    'transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm':
        KernelChoice(op='rms_norm', backends=('liger', 'npu')),
})
```

`DEFAULT_KERNEL_CONFIG` 本身不应在运行时原地修改。

**日志分级**：`mapping=None`（默认配置路径）时，回退/失败日志全部是 DEBUG（CUDA 机器、家族缺失是常态）；显式传入 mapping（哪怕是默认配置的复制合并）则升为 WARNING——明确表达了意图，没生效必须告知。每次成功安装有一条 INFO：

```text
[kernelize] target=...Qwen3MLP.forward op=swiglu backend=npu installer=default
[kernelize] target=sdpa op=sdpa_attention backend=npu installer=install_sdpa
```

**CUDA 行为**：默认配置跨平台统一。CUDA 上 `('npu', 'liger')` 链中 npu 的 `available()` 为 False 自动落到 liger；liger 未装则整链失败（默认配置下 DEBUG 跳过 ≈ no-op）。

## 场景示例

### 按算子点名选实现（带替补顺序）

```python
model = kernelize(model, {**DEFAULT_KERNEL_CONFIG,
    'transformers.models.qwen3.modeling_qwen3.Qwen3MLP.forward':
        KernelChoice(op='swiglu', backends=('liger', 'npu')),   # 首选 liger，不行退 npu
})
```

### 字符串 key + 直接 impl（绕过选择）

```python
from twinkle.kernel import kernelize
from twinkle.kernel.ops.rotary.npu import npu_apply_rotary_pos_emb

model = kernelize(model, {
    'transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb':
        npu_apply_rotary_pos_emb,
})
```

### 自定义类替换

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

model = kernelize(model, {Qwen2RMSNorm: MyRMSNorm})
```

### Hub Kernel（HF Hub 格式）

```python
from twinkle.kernel import hub, kernelize

model = kernelize(model, {
    SiluAndMul: hub('kernels-community/activation:SiluAndMul', version=1),
})
```

`revision` 与 `version` 二选一必传。`hub(...)` 触发 `kernels` 包的延迟 import，未安装时会提示 `pip install kernels`。

### GMM（grouped matmul）opt-in

默认配置**不包含** `transformers.integrations.moe._grouped_mm` 的 NPU 替换（没有 Expert Parallelism 时约 8x 开销）。需要时显式加入（注意：直接 impl 无平台门控，仅在确认 NPU 环境时使用）：

```python
from twinkle.kernel.ops.moe.npu import npu_grouped_mm

model = kernelize(model, {**DEFAULT_KERNEL_CONFIG,
    'transformers.integrations.moe._grouped_mm': npu_grouped_mm,
})
```

## op 注册机制（如何新增 op / backend）

内置实现按算子组织在 `twinkle/kernel/ops/<op>/` 下：`__init__.py` 负责注册（只含惰性引用与轻量可用性检查，**不得直接 import 可选依赖**），`<backend>.py` 是实现本体。注册 = 登记"某 backend 能给某 op 提供实现"：

```python
# twinkle/kernel/ops/swiglu/__init__.py
from ...registry import KernelImpl, is_liger_available, is_npu_available, lazy_import, register_op

register_op(
    'swiglu',
    implementations={
        'npu': KernelImpl(
            load=lazy_import('twinkle.kernel.ops.swiglu.npu:npu_swiglu_forward'),
            available=is_npu_available,
        ),
        'liger': KernelImpl(
            load=lazy_import('twinkle.kernel.ops.swiglu.liger:liger_swiglu_forward'),
            available=is_liger_available,
        ),
    },
)
```

- `KernelImpl.load(target)`：惰性加载工厂，仅在实现被选中时调用；接收 mapping target，可按目标家族特化（如 liger 的 RMSNorm 按 gemma / qwen3_5 变体分派）
- `KernelImpl.available()`：返回 `(True, None)` 可用 / `(False, reason)` 不可用并顺延；平台与依赖判断都收在这里
- 同名 op 重复注册、空 implementations → `ValueError`
- **vendor 规范**：impl 引入外部 kernel 时，文件 docstring 必须注明来源仓库@commit、本地改动清单，以及 re-sync 提醒

### 自定义 installer

普通 class / attr 替换之外的安装方式，由 op 自带 installer（签名 `(model, target, impl)`）。优先级：`KernelChoice.installer` → `OpDefinition.installer` → 通用 installer。例子——SDPA 写入 transformers 全局 attention 注册表：

```python
def install_sdpa(model, target, impl) -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    AttentionInterface._global_mapping['sdpa'] = impl
    ALL_ATTENTION_FUNCTIONS['sdpa'] = impl

register_op('sdpa_attention', implementations={...}, installer=install_sdpa)
```

默认配置用逻辑 target 引用这类 op（`'sdpa': KernelChoice(op='sdpa_attention', backends=('npu',))`）；逻辑 target 仅作标识传给 installer，不由通用替换器解析。installer 执行失败会上抛异常——不让模型处于用户不知情的半安装状态。

## 迁移指南（旧接口 → 新写法）

| 旧用法 | 新写法 |
| --- | --- |
| `kernelize(model, npu_builtin(model))` | `kernelize(model)`（默认配置 NPU 上 CANN 优先） |
| `kernelize(model, liger_builtin(model))` | 自建全 liger mapping，或 `{**DEFAULT_KERNEL_CONFIG, ...}` 把想改的 target 链写成 `('liger', ...)` |
| `{**npu_builtin(m), **liger_builtin(m)}` 手动合并 | `{**DEFAULT_KERNEL_CONFIG, ...}` 按 target 覆盖 |
| `{Qwen3RMSNorm: {'npu': NpuRMSNorm}}`（设备条件 dict） | `{Qwen3RMSNorm: NpuRMSNorm}` 或 `KernelChoice(op='rms_norm', backends=('npu',))` |
| `from twinkle.kernel.npu_impls.x import ...` | `from twinkle.kernel.ops.<op>.npu import ...` |
| `from twinkle.kernel.liger_impls.x import ...` | `from twinkle.kernel.ops.<op>.liger import ...` |

`npu_builtin()` / `liger_builtin()` / 设备条件 dict（`{'npu': impl}`）已删除，无兼容层；平台判断收进 `KernelImpl.available()`。NPU 上 `kernelize(model)` 的默认替换结果与旧版一致；CUDA 上从 no-op 变为应用默认配置（liger 可用即生效，未装 ≈ no-op）。

## 环境变量

只有两个保留：

- `TWINKLE_NPU_FLA`：Qwen3.5 FLA 开关（默认开，设为 `0`/`false` 关闭）
- `TWINKLE_NPU_GATED_RMSNorm_FP32`：将 Gated RMSNorm 强制升到 FP32 计算（默认关）

## 注意事项

- `m.__class__ = impl_cls` 是 Python class 替换魔法。impl 类**必须**只覆盖 `forward`（以及辅助方法），不能定义 `__init__`，否则原 instance 的 attribute 会与 impl 的预期错位
- 精确匹配：`type(m) is target_cls`。继承自 `target_cls` 的子类不会被替换；如需替换，把子类也放进 mapping
- 调用 `kernelize` 多次是幂等的（`__class__` 已是 impl 时再设一次无害）
- 没有 `unkernelize`——替换是单向的
