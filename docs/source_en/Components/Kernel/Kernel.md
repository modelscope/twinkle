# Twinkle Kernel

`twinkle.kernel` exposes a mapping-driven kernel replacement API. Replacing one
implementation with another collapses to a single `kernelize(model, mapping)`
call.

The public surface is exactly three symbols:

| Symbol | Purpose |
| --- | --- |
| `kernelize(model, mapping=None)` | Apply ``mapping`` to ``model`` (in place) and return it. If ``mapping`` is omitted, it is auto-detected from the current platform (see below) |
| `npu_builtin(model=None)` | Return the Ascend NPU built-in mapping (composes with user mappings) |
| `hub(ref, *, revision=None, version=None, backend=None, trust_remote_code=False)` | Build a ``HubRef`` for use as a mapping value; the actual Hub download is deferred to ``kernelize`` |

## Mapping semantics

`mapping` keys describe the target to replace:

- `type[nn.Module]` subclass — replace **every** instance whose exact type matches (`m.__class__ = impl`; subclasses are **not** touched)
- `str` of the form `'pkg.sub.attr'` or `'pkg.sub.ClassName.attr'` — `setattr(target, attr, impl)`

`mapping` values describe the replacement:

- `type[nn.Module]` subclass — used as the impl class. The class' `__init__` is **never** invoked; its forward must work against the attributes the original instance already has
- `Callable` — assigned with `setattr`
- `dict[str, V]` — device → impl dispatch. Device is inferred from the model; entries without a matching key are **silently skipped**
- `HubRef` — built via `hub(...)`; resolved lazily

Device is inferred from `next(model.parameters()).device.type` (falling back to buffers, then `'cpu'`).

## Auto-detection (mapping omitted)

When `mapping` is `None`, `kernelize` auto-detects the current platform via `Platform.device_prefix()` and applies the matching built-in bundle. Platforms without a built-in bundle are a safe no-op (the model is returned unchanged).

## Examples

### Enable the built-in bundle for the current platform

```python
from twinkle.kernel import kernelize

model = kernelize(model)  # auto-detects the platform and applies its built-in bundle
```

The explicit form is still supported:

```python
import torch
from twinkle.kernel import kernelize, npu_builtin

if torch.npu.is_available():
    model = kernelize(model, npu_builtin(model))
```

### Custom class replacement

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from twinkle.kernel import kernelize

model = kernelize(model, {Qwen2RMSNorm: MyRMSNorm})
```

### Built-in + custom override

```python
from twinkle.kernel import kernelize, npu_builtin

model = kernelize(model, {**npu_builtin(model), Qwen2RMSNorm: MyRMSNorm})
```

Plain dict merge — later keys override earlier ones.

### Hub kernel (HF Hub format)

```python
from twinkle.kernel import kernelize, hub
from my_pkg import SiluAndMul

model = kernelize(model, {
    SiluAndMul: hub('kernels-community/activation:SiluAndMul', version=1),
})
```

Exactly one of `revision` / `version` must be passed. The `kernels` package is imported lazily; absence raises a clear "install kernels" error.

### Function-level replacement

```python
from twinkle.kernel import kernelize
from twinkle.kernel.npu_impls.rotary import npu_apply_rotary_pos_emb

model = kernelize(model, {
    'transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb':
        npu_apply_rotary_pos_emb,
})
```

### Cross-device mapping (NPU enabled, CUDA skipped)

```python
from twinkle.kernel import kernelize

model = kernelize(model, {
    Qwen2RMSNorm: {'npu': NpuRMSNorm, 'cuda': CudaRMSNorm},
})
```

Safe to run on CUDA — entries whose dict misses the current device just skip.

## NPU built-in coverage

`npu_builtin(model)` returns a dict that (as available transformers modules permit) covers:

- RMSNorm class replacement for Qwen2 / Qwen3 / Qwen3-MoE / Qwen2.5-VL / Qwen3.5 / Qwen3.5-MoE families
- `apply_rotary_pos_emb` function replacement (fused RoPE) for the same families
- SwiGLU fused replacement for the MLP variants
- `Experts.forward` and `SparseMoeBlock.forward` for Qwen3-MoE / Qwen3.5-MoE
- GatedRMSNorm forward for Qwen3.5 / Qwen3.5-MoE
- `apply_multimodal_rotary_pos_emb` for Qwen2.5-VL
- Global SDPA replacement (one-shot side effect on `ALL_ATTENTION_FUNCTIONS['sdpa']`)
- Qwen3.5 Flash Linear Attention enablement (one-shot side effect + per-instance traversal, triggered inside `npu_builtin(model)`)

**Not included by default:** the NPU replacement for `transformers.integrations.moe._grouped_mm`. Without Expert Parallelism the contiguous-copy overhead is ~8x. Opt in explicitly when EP is enabled:

```python
from twinkle.kernel import kernelize, npu_builtin
from twinkle.kernel.npu_impls.moe import npu_grouped_mm

mapping = {
    **npu_builtin(model),
    'transformers.integrations.moe._grouped_mm': {'npu': npu_grouped_mm},
}
model = kernelize(model, mapping)
```

## Environment variables

Only two remain:

- `TWINKLE_NPU_FLA` — Qwen3.5 FLA switch (default on; `0`/`false` to disable)
- `TWINKLE_NPU_GATED_RMSNorm_FP32` — force FP32 in Gated RMSNorm forward (default off)

The legacy `TWINKLE_NPU_PATCH` / `TWINKLE_NPU_FUSED_OPS` / `TWINKLE_NPU_GMM_PATCH` / `TWINKLE_USE_KERNELS` are gone — they're now "include the entry in the mapping or don't" decisions.

## Caveats

- `m.__class__ = impl_cls` is Python class-replacement magic. The impl class **must** override only `forward` (and helpers); defining `__init__` is incompatible with the contract
- Exact match: `type(m) is target_cls`. Subclasses of `target_cls` are not replaced — add them to the mapping yourself
- `kernelize` is idempotent under repeated calls
- There is no `unkernelize` — replacement is one-way
