# DSV4 EP Multi-LoRA Target Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `MultiLoraTransformersModel` 中支持 DSV4 EP SFT 训练常驻多个 target-parameters LoRA adapter，并保证每次训练只激活一个 adapter，保存结果可被原始 Transformers + PEFT `target_parameters` 推理路径直接加载。

**Architecture:** 保留 DSV4 packed expert `nn.Parameter` 结构，不拆成 `nn.Linear`。在 Twinkle `MultiLora` 中新增 target-parameter multi-slot wrapper，复用现有 tenant adapter 到物理 slot 的映射；普通 `target_modules` LoRA 继续走现有 PEFT slot path，裸参数 LoRA 走新 wrapper path。

**Tech Stack:** Python, PyTorch `nn.utils.parametrize`, PEFT `LoraConfig`, Twinkle `MultiLoraTransformersModel`, native FSDP2, pytest.

---

## 设计输入

- Spec: `docs/superpowers/specs/2026-06-01-dsv4-ep-multi-lora-target-parameters-design.md`
- 相关上游语义参考：PEFT `peft/tuners/lora/layer.py::ParamWrapper`
- 当前 multi-LoRA 入口：
  - `src/twinkle/model/multi_lora.py`
  - `src/twinkle/model/transformers/multi_lora_transformers.py`
  - `src/twinkle/model/transformers/strategy/native_fsdp.py`

## 文件结构

- Create: `src/twinkle/model/multi_lora_target_parameters.py`
  - 负责 target-parameter LoRA wrapper、PEFT-compatible delta math、slot activation、state dict key normalization。
- Modify: `src/twinkle/model/multi_lora.py`
  - 接入 target-parameter manager；扩展 `activate_adapter`、`deactivate_adapter`、`save_initial_weights`、`set_state_dict`、`get_state_dict`、统计和 trainable 参数示例。
- Modify: `src/twinkle/model/transformers/multi_lora_transformers.py`
  - 在 adapter config 包含 `target_parameters` 时安装 target-parameter slots；保证 EP 在安装前应用，FSDP wrap 在安装后发生。
- Modify: `src/twinkle/model/transformers/strategy/native_fsdp.py`
  - 只在现有 full-state gather 无法覆盖新 wrapper key 时修改；优先通过 wrapper key 命名复用已有 `_ep_expert_state_dict_gather_dim`。
- Create: `tests/model/test_multi_lora_target_parameters.py`
  - 覆盖 fake 3D expert parameter 的 delta math、单 active adapter 更新、state dict save/load、PEFT 兼容 key/shape。
- Modify or Create: `tests/moe/test_ep_multi_lora_target_parameters.py`
  - 可选/慢测试，验证 EP/FSDP SFT 路径下两个 adapters 常驻、交替训练、分别保存。
- Modify: `cookbook/transformers/ep_fsdp2_lora_deepseek_v4.py`
  - 可选更新：增加 `USE_MULTI_LORA_MODEL=1` 或新增 SFT 示例脚本。实现功能通过测试验证后再做。

---

### Task 1: 锁定 PEFT ParamWrapper 的 checkpoint key/shape

**Files:**
- Create: `tests/model/test_multi_lora_target_parameters.py`

- [x] **Step 1: 写 PEFT characterization 测试**

在 `tests/model/test_multi_lora_target_parameters.py` 中先写一个只依赖 PEFT 的 fake module，记录 `target_parameters` 对 3D expert 参数生成的 key 和 shape。

```python
import torch
from torch import nn
from peft import LoraConfig, get_peft_model


class FakePackedExperts(nn.Module):
    def __init__(self, num_experts=2, hidden=4, intermediate=6, *, is_transposed=False):
        super().__init__()
        self.is_transposed = is_transposed
        if is_transposed:
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, intermediate * 2, hidden))
            self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
        else:
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate * 2))
            self.down_proj = nn.Parameter(torch.randn(num_experts, intermediate, hidden))

    def forward(self, x, expert_idx=0):
        gate_up = self.gate_up_proj[expert_idx]
        down = self.down_proj[expert_idx]
        if self.is_transposed:
            hidden = torch.nn.functional.linear(x, gate_up)
            gate, up = hidden.chunk(2, dim=-1)
            return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down)
        hidden = torch.nn.functional.linear(x, gate_up.T)
        gate, up = hidden.chunk(2, dim=-1)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down.T)


class FakeModel(nn.Module):
    def __init__(self, *, is_transposed=False):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = FakePackedExperts(is_transposed=is_transposed)

    def forward(self, x, expert_idx=0):
        return self.mlp.experts(x, expert_idx=expert_idx)


def test_peft_target_parameter_key_shapes_for_3d_experts():
    model = FakeModel()
    cfg = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=[],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )

    peft_model = get_peft_model(model, cfg, adapter_name="default")
    state = peft_model.state_dict()
    lora_keys = sorted(k for k in state if "lora_" in k)

    assert lora_keys
    assert any("lora_A.default.weight" in k for k in lora_keys)
    assert any("lora_B.default.weight" in k for k in lora_keys)
    assert all(state[k].ndim == 2 for k in lora_keys)
```

- [x] **Step 2: 运行 characterization 测试，确认 PEFT 当前行为**

Run: `pytest tests/model/test_multi_lora_target_parameters.py::test_peft_target_parameter_key_shapes_for_3d_experts -q`

Expected: PASS。若失败，先根据 PEFT 0.19.1 实际 key/shape 调整测试，让它成为后续实现的兼容契约。

- [x] **Step 3: 提交 characterization 测试**

```bash
git add tests/model/test_multi_lora_target_parameters.py
git commit -m "test: characterize peft target parameter keys"
```

---

### Task 2: 写 Twinkle target-parameter multi-slot 的失败测试

**Files:**
- Modify: `tests/model/test_multi_lora_target_parameters.py`

- [x] **Step 1: 写两个 tenant adapter 交替训练的失败测试**

追加测试，先假设存在 `TargetParameterLoraManager`。

```python
from peft import LoraConfig
from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager


def _make_target_cfg(r=2):
    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=[],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )


def test_target_parameter_multi_lora_updates_only_active_adapter():
    torch.manual_seed(0)
    model = FakeModel()
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    manager.patch(model, target_parameters=_make_target_cfg().target_parameters)
    manager.acquire("adapter_a", "lora_0", _make_target_cfg(r=2))
    manager.acquire("adapter_b", "lora_1", _make_target_cfg(r=2))

    params_before = {
        name: param.detach().clone()
        for name, param in manager.named_slot_parameters("adapter_b")
    }

    opt = torch.optim.SGD(manager.parameters_for_tenant("adapter_a"), lr=0.1)
    with manager.adapter("adapter_a"):
        loss = model(torch.randn(3, 4), expert_idx=0).pow(2).mean()
    loss.backward()
    opt.step()

    for name, param in manager.named_slot_parameters("adapter_b"):
        assert torch.equal(param.detach(), params_before[name])
```

- [x] **Step 2: 运行测试，确认因为模块不存在而失败**

Run: `pytest tests/model/test_multi_lora_target_parameters.py::test_target_parameter_multi_lora_updates_only_active_adapter -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'twinkle.model.multi_lora_target_parameters'`.

- [x] **Step 3: 提交失败测试**

```bash
git add tests/model/test_multi_lora_target_parameters.py
git commit -m "test: cover target parameter multi lora activation"
```

---

### Task 3: 实现 target-parameter wrapper 和 manager

**Files:**
- Create: `src/twinkle/model/multi_lora_target_parameters.py`
- Modify: `tests/model/test_multi_lora_target_parameters.py`

- [x] **Step 1: 新建实现文件骨架**

Create `src/twinkle/model/multi_lora_target_parameters.py`，至少包含：

```python
from __future__ import annotations

import math
import re
import torch
from contextlib import contextmanager
from dataclasses import dataclass
from peft import LoraConfig
from torch import nn
from typing import Iterable, Iterator


class LoraParameterProxy(nn.Module):
    def __init__(self, delta_weight: torch.Tensor):
        super().__init__()
        self.delta_weight = delta_weight

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + self.delta_weight.to(device=weight.device)


@dataclass
class TargetParameterRecord:
    module_name: str
    module: nn.Module
    parameter_name: str


class TargetParameterLoraWrapper(nn.Module):
    def __init__(self, record: TargetParameterRecord, max_loras: int, max_r: int):
        super().__init__()
        self.record = record
        self.max_loras = max_loras
        self.max_r = max_r
        self.active_adapter: str | None = None
        self.disable_adapters = False
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.scaling: dict[str, float] = {}
        self.r: dict[str, int] = {}
        self._init_slots()

    def _init_slots(self) -> None:
        # Implement after feature tests lock expected shapes.
        raise NotImplementedError

    def configure_slot(self, slot_name: str, config: LoraConfig) -> None:
        raise NotImplementedError

    def get_delta_weight(self, slot_name: str) -> torch.Tensor:
        raise NotImplementedError

    @contextmanager
    def activate(self, slot_name: str | None):
        raise NotImplementedError
```

Manager API:

```python
class TargetParameterLoraManager:
    def __init__(self, max_loras: int, max_r: int):
        self.max_loras = max_loras
        self.max_r = max_r
        self.wrappers: list[TargetParameterLoraWrapper] = []
        self.tenant_to_slot: dict[str, str] = {}
        self.tenant_configs: dict[str, LoraConfig] = {}

    def patch(self, model: nn.Module, target_parameters: Iterable[str]) -> None:
        raise NotImplementedError

    def acquire(self, tenant_adapter_name: str, slot_name: str, config: LoraConfig) -> None:
        raise NotImplementedError

    @contextmanager
    def adapter(self, tenant_adapter_name: str, disable_lora: bool = False):
        raise NotImplementedError

    def parameters_for_tenant(self, tenant_adapter_name: str) -> list[nn.Parameter]:
        raise NotImplementedError
```

- [x] **Step 2: 实现目标参数匹配**

`patch()` 遍历 `model.named_modules()` 和 `module.named_parameters(recurse=False)`，用 PEFT 同样的 suffix 规则匹配：

```python
key = f"{module_name}.{param_name}" if module_name else param_name
if key in target_parameters or any(key.endswith(f".{target}") for target in target_parameters):
    ...
```

Expected behavior:
- 找不到任何目标参数时抛 `ValueError("target_parameters=... were set but no parameter was matched")`。
- 参数维度不是 2 或 3 时抛 `ValueError`。

- [x] **Step 3: 实现 slot shape 和 delta math**

按照 PEFT `ParamWrapper`：
- 2D: `lora_A[slot] = nn.Linear(in_features, max_r, bias=False)`，`lora_B[slot] = nn.Linear(max_r, out_features, bias=False)`。
- 3D: `lora_A[slot] = nn.Linear(in_features, max_r * num_experts, bias=False)`，`lora_B[slot] = nn.Linear(max_r * num_experts, out_features, bias=False)`。
- `config.r` 只使用前 `r` rank。
- 对 3D 使用：

```python
weight_A = lora_A.weight[: r * num_experts].reshape(num_experts, r, in_features)
weight_B = lora_B.weight[:, : r * num_experts].reshape(out_features, r, num_experts)
if not did_swap_in_out_features:
    delta = torch.einsum("o r e, e r i -> e i o", weight_B, weight_A) * scaling
else:
    delta = torch.einsum("o r e, e r i -> e o i", weight_B, weight_A) * scaling
```

- [x] **Step 4: 实现临时 parametrization**

`TargetParameterLoraWrapper.activate(slot_name)` 使用：

```python
nn.utils.parametrize.register_parametrization(
    record.module,
    record.parameter_name,
    LoraParameterProxy(delta_weight),
)
try:
    with nn.utils.parametrize.cached():
        yield
finally:
    nn.utils.parametrize.remove_parametrizations(
        record.module,
        record.parameter_name,
        leave_parametrized=False,
    )
```

如果 `disable_lora=True` 或 slot 不存在，则直接 `yield`。

- [x] **Step 5: 运行 Task 2 测试，确认通过**

Run: `pytest tests/model/test_multi_lora_target_parameters.py::test_target_parameter_multi_lora_updates_only_active_adapter -q`

Expected: PASS。

- [x] **Step 6: 提交 wrapper/manager 基础实现**

```bash
git add src/twinkle/model/multi_lora_target_parameters.py tests/model/test_multi_lora_target_parameters.py
git commit -m "feat: add target parameter multi lora manager"
```

---

### Task 4: 接入 MultiLora 的 tenant slot 生命周期

**Files:**
- Modify: `src/twinkle/model/multi_lora.py`
- Modify: `tests/model/test_multi_lora_target_parameters.py`

- [x] **Step 1: 为 MultiLora 增加 target manager 字段**

在 `MultiLora.__init__` 增加：

```python
self.target_parameter_manager = TargetParameterLoraManager(max_loras=max_loras, max_r=max_r)
```

- [x] **Step 2: 增加安装入口**

添加方法：

```python
def patch_target_parameters(self, module, target_parameters):
    self.target_parameter_manager.patch(module, target_parameters)
```

- [x] **Step 3: 在 acquire_lora 中配置 target-parameter slot**

当 `config.target_parameters` 非空时：

```python
self.target_parameter_manager.acquire(
    tenant_adapter_name=tenant_adapter_name,
    slot_name=_available_lora.adapter_name,
    config=config,
)
```

- [x] **Step 4: 扩展 adapter context**

修改 `MultiLora.adapter()`：

```python
with self.target_parameter_manager.adapter(tenant_adapter_name, disable_lora=disable_lora):
    yield self.find_lora_by_tenant(tenant_adapter_name).adapter_name
```

确保 regular module 的 `activate_adapter/deactivate_adapter` 行为不变。

- [x] **Step 5: 扩展 release/reset 初始权重**

`release_lora()` 释放 tenant 时同时调用：

```python
self.target_parameter_manager.release(tenant_adapter_name)
```

并补测试验证释放后 slot B 权重置零、A 恢复初始值。

- [x] **Step 6: 运行单元测试**

Run: `pytest tests/model/test_multi_lora_target_parameters.py -q`

Expected: PASS。

- [x] **Step 7: 提交 MultiLora 生命周期接入**

```bash
git add src/twinkle/model/multi_lora.py tests/model/test_multi_lora_target_parameters.py
git commit -m "feat: integrate target parameter slots with multilora"
```

---

### Task 5: 实现 state_dict 保存/加载和 PEFT 兼容性

**Files:**
- Modify: `src/twinkle/model/multi_lora_target_parameters.py`
- Modify: `src/twinkle/model/multi_lora.py`
- Modify: `tests/model/test_multi_lora_target_parameters.py`

- [x] **Step 1: 为 manager 增加导出 API**

在 `TargetParameterLoraManager` 增加：

```python
def get_state_dict(self, tenant_adapter_name: str) -> dict[str, torch.Tensor]:
    ...

def set_state_dict(self, tenant_adapter_name: str, state_dict: dict[str, torch.Tensor]) -> None:
    ...

def named_slot_parameters(self, tenant_adapter_name: str) -> Iterator[tuple[str, nn.Parameter]]:
    ...
```

导出时使用 PEFT-compatible logical key，去掉物理 slot 名：

```python
model.layers.0.mlp.experts.lora_A.weight
model.layers.0.mlp.experts.lora_B.weight
```

具体 key 必须以 Task 1 characterization 结果为准。

- [x] **Step 2: rank slice 和 padding**

导出只保留 tenant config 的实际 `r`：

```python
A = A[: r * num_experts, :]
B = B[:, : r * num_experts]
```

加载时把 checkpoint 中实际 rank 拷入 `max_r` slot，剩余部分置零。

- [x] **Step 3: 扩展 MultiLora get/set_state_dict**

`MultiLora.get_state_dict()` 返回 regular module LoRA 与 target-parameter LoRA 的合并结果。key 冲突时抛错：

```python
state = regular_state
target_state = self.target_parameter_manager.get_state_dict(tenant_adapter_name)
overlap = state.keys() & target_state.keys()
if overlap:
    raise ValueError(f"Duplicate LoRA state keys: {sorted(overlap)[:5]}")
state.update(target_state)
```

`set_state_dict()` 同理：regular key 走现有逻辑，target key 交给 manager；未知 key 最后抛错。

- [x] **Step 4: 写 PEFT 加载兼容测试**

在 fake model 上：
1. 用 Twinkle manager 训练/手动改 adapter A 权重。
2. `state = manager.get_state_dict("adapter_a")`。
3. 新建未 patch 的 `FakeModel()`。
4. 用 `get_peft_model(... LoraConfig(target_parameters=...))`。
5. 调 `peft.utils.set_peft_model_state_dict(peft_model, state, adapter_name="default")`。
6. 比较两个模型同输入输出一致。

- [x] **Step 5: 运行保存/加载兼容测试**

Run: `pytest tests/model/test_multi_lora_target_parameters.py::test_target_parameter_state_dict_loads_with_peft -q`

Expected: PASS。

- [x] **Step 6: 提交 state_dict 支持**

```bash
git add src/twinkle/model/multi_lora_target_parameters.py src/twinkle/model/multi_lora.py tests/model/test_multi_lora_target_parameters.py
git commit -m "feat: export target parameter lora checkpoints"
```

---

### Task 6: 接入 MultiLoraTransformersModel 和 EP 注入顺序

**Files:**
- Modify: `src/twinkle/model/transformers/multi_lora_transformers.py`
- Modify: `tests/model/test_multi_lora_target_parameters.py`

- [x] **Step 1: 增加 target-parameter 安装 helper**

在 `MultiLoraTransformersModel` 中增加：

```python
def _ensure_target_parameter_lora_installed(self, config: LoraConfig) -> None:
    target_parameters = getattr(config, "target_parameters", None)
    if not target_parameters:
        return
    if self._model_wrapped:
        raise RuntimeError("target_parameters LoRA must be installed before FSDP/DDP wrapping")
    if getattr(self, "_enable_expert_parallel", False):
        self.strategy.capture_pre_ep_state_if_needed(self.model, enable_ep=True)
        self._maybe_apply_expert_parallel()
    self.multi_adapter.patch_target_parameters(self.model, target_parameters)
```

如果 target manager 已经安装过同一组 target parameters，应跳过重复安装；如果新 config 请求不同 target set，报错或只安装新增 target，优先选择报错以降低复杂度。

- [x] **Step 2: 在 add_adapter_to_model 中调用 helper**

在 `add_adapter_to_model()` 验证 `LoraConfig` 后、`self.multi_adapter.acquire_lora(...)` 前调用：

```python
self._ensure_target_parameter_lora_installed(config_or_dir)
```

- [x] **Step 3: 修正 `_get_trainable_parameters`**

当前实现只通过 `super()._get_trainable_parameters(real_adapter_name)` 按 `.lora_*.slot.` 正则抓 regular LoRA。扩展为：

```python
params = super()._get_trainable_parameters(real_adapter_name)
params.update(self.multi_adapter.get_target_parameter_trainable_parameters(adapter_name))
return params
```

确保 optimizer 只拿当前 tenant 的 target-parameter slot。

- [x] **Step 4: 写模型级测试**

用 `FakeModel` 直接构造 `MultiLora` 或轻量 `MultiLoraTransformersModel` 替身，验证：
- 第一个 adapter 安装 target slots。
- 第二个 adapter 复用已有 target slots。
- `_get_trainable_parameters("adapter_a")` 不包含 `adapter_b` 的 target LoRA 参数。

- [x] **Step 5: 运行模型级测试**

Run: `pytest tests/model/test_multi_lora_target_parameters.py -q`

Expected: PASS。

- [x] **Step 6: 提交 Transformers 接入**

```bash
git add src/twinkle/model/transformers/multi_lora_transformers.py tests/model/test_multi_lora_target_parameters.py
git commit -m "feat: enable target parameters in multilora transformers"
```

---

### Task 7: 验证 native FSDP full-state gather 与 EP checkpoint

**Files:**
- Modify: `src/twinkle/model/transformers/strategy/native_fsdp.py` if needed
- Create: `tests/moe/test_ep_multi_lora_target_parameters.py`

- [x] **Step 1: 写慢测试骨架**

创建 `tests/moe/test_ep_multi_lora_target_parameters.py`，沿用 `tests/moe/test_ep_fsdp_vs_single.py` 的 skip 条件：

```python
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason="Need 4 GPUs")
def test_ep_fsdp_multi_lora_target_parameter_checkpoint_smoke():
    ...
```

测试目标仅覆盖 SFT 训练语义：
- 构造 EP/FSDP mesh。
- 创建包含 3D packed experts 的 fake MoE 或使用可用小 MoE 模型。
- 添加 `adapter_a` 和 `adapter_b`。
- 交替各训练一步。
- 保存 `adapter_a` 和 `adapter_b`。
- 用未修改的 PEFT target-parameters 路径加载保存结果。

不在本测试中覆盖 sampler、vLLM、权重同步、RL/GRPO 或 rollout。

- [x] **Step 2: 先运行慢测试，观察失败点**

Run: `pytest tests/moe/test_ep_multi_lora_target_parameters.py -q -s`

Expected: 初次可能 FAIL，常见失败包括 full-state key 没有 gather、lora_A/B gather dim 不对、保存 key 与 PEFT 不一致。

- [x] **Step 3: 如有必要，调整 native FSDP gather**

优先让 target wrapper 的 `named_parameters()` key 包含 `lora_A` / `lora_B` 并位于 `_detect_ep_expert_names()` 能识别的 expert module 下。只有这个方式不可行时才修改：

- `_detect_ep_expert_names`
- `_ep_expert_state_dict_gather_dim`
- `_collect_adapter_source_state`
- `_split_for_ep_pre_distribute`

新增分支必须有测试覆盖。

- [x] **Step 4: 运行快测和慢测**

Run:

```bash
pytest tests/model/test_multi_lora_target_parameters.py -q
pytest tests/moe/test_ep_multi_lora_target_parameters.py -q -s
```

Expected: model tests PASS；EP smoke 在满足 GPU/model 条件时 PASS，否则按 skip 条件 SKIPPED。

- [x] **Step 5: 提交 EP/FSDP 验证**

```bash
git add src/twinkle/model/transformers/strategy/native_fsdp.py tests/moe/test_ep_multi_lora_target_parameters.py
git commit -m "test: cover ep fsdp target parameter multi lora"
```

---

### Task 8: SFT Cookbook 和用户入口

**Files:**
- Modify: `cookbook/transformers/ep_fsdp2_lora_deepseek_v4.py` or Create: `cookbook/transformers/ep_fsdp2_multi_lora_deepseek_v4.py`
- Modify: `cookbook/transformers/ep_fsdp2_lora_deepseek_v4.sh` if reusing existing script

- [x] **Step 1: 选择最小入口**

优先新增 `cookbook/transformers/ep_fsdp2_multi_lora_deepseek_v4.py`，避免改变现有单 adapter cookbook 默认行为。该示例只展示 SFT 训练，不接 sampler/vLLM，也不覆盖 RL/GRPO。

- [x] **Step 2: 示例中使用 MultiLoraTransformersModel**

示例展示：

```python
from twinkle.model import MultiLoraTransformersModel

model = MultiLoraTransformersModel(
    model_id=MODEL_ID,
    config=config,
    device_mesh=device_mesh,
    strategy="native_fsdp",
    memory_efficient_init=True,
    max_loras=MAX_LORAS,
    max_r=MAX_R,
    fsdp_config={
        "expert_parallel": {
            "enabled": ENABLE_EP,
            "router_dtype": "fp32",
            "keep_router_logits": False,
        }
    },
)
```

添加两个 adapter：

```python
for adapter_name in ADAPTER_NAMES:
    model.add_adapter_to_model(adapter_name, lora_cfg, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    model.set_optimizer("AdamW", lr=LR, foreach=False, adapter_name=adapter_name)
```

训练循环中显式传 `adapter_name=current_adapter`。

- [x] **Step 3: 运行语法检查**

Run: `python -m py_compile cookbook/transformers/ep_fsdp2_multi_lora_deepseek_v4.py`

Expected: PASS。

- [x] **Step 4: 提交 cookbook**

```bash
git add cookbook/transformers/ep_fsdp2_multi_lora_deepseek_v4.py
git commit -m "docs: add dsv4 ep multi lora cookbook"
```

---

### Task 9: 最终验证和清理

**Files:**
- All modified files

- [x] **Step 1: 运行目标快测**

Run: `pytest tests/model/test_multi_lora_target_parameters.py -q`

Expected: PASS。

- [x] **Step 2: 运行相关 EP 测试**

Run: `pytest tests/moe/test_ep_multi_lora_target_parameters.py -q -s`

Expected: PASS if environment has required GPUs/model; otherwise SKIPPED with explicit reason.

- [x] **Step 3: 运行现有 EP 回归测试**

Run: `pytest tests/moe/test_ep_fsdp_vs_single.py -q -s`

Expected: PASS if environment has required GPUs/model; otherwise SKIPPED with explicit reason.

- [x] **Step 4: 检查 diff**

Run: `git diff --stat HEAD`

Expected: Only target-parameter multi-LoRA implementation, tests, and optional cookbook/docs are present.

- [x] **Step 5: 更新计划勾选状态**

在本文件中勾选已完成步骤，保持执行记录可追踪。

- [x] **Step 6: 最终提交**

```bash
git add docs/superpowers/plans/2026-06-01-dsv4-ep-multi-lora-target-parameters.md
git commit -m "docs: add dsv4 ep multi lora implementation plan"
```

---

## 执行注意事项

- 不要修改 PEFT site-packages。
- 不要把 DSV4 packed expert 转成 `nn.Linear`。
- 不要让同一次 forward 叠加多个 tenant adapter；本计划只支持一次激活一个。
- 本轮验收范围只包括 SFT 训练；不要实现或验证 sampler、vLLM、权重同步、RL/GRPO 或 rollout。
- `target_parameters` 的保存 key/shape 以 Task 1 characterization 测试为准。
- 如果 PEFT 未来支持 target-parameters multi-adapter，优先让新增实现保持隔离，方便替换。
- 每个任务完成后提交一次，小步推进，避免把 checkpoint、EP、activation 问题混在一个大 diff 里。
