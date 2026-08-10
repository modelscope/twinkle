# Twinkle Kernel

`twinkle.kernel` exposes a mapping-driven kernel replacement API: swapping one
implementation for another in a model collapses to a single
`kernelize(model, mapping)` call. Three concerns are fully separated: **what to
replace** (mapping key), **which implementation to pick** (`KernelChoice`), and
**how to install it** (installer).

The public surface is exactly four symbols:

| Symbol | Purpose |
| --- | --- |
| `kernelize(model, mapping=None)` | Apply `mapping` to `model` in place and return it. If `mapping` is omitted, the built-in `DEFAULT_KERNEL_CONFIG` is applied |
| `KernelChoice(op, backends, installer=None)` | Mapping value: which op to use for this replacement, and in what priority order to pick a backend |
| `DEFAULT_KERNEL_CONFIG` | Built-in default mapping (target → `KernelChoice`); copy/merge/override it to customize |
| `hub(ref, *, revision=None, version=None, backend=None)` | Build a `HubRef` for use as a mapping value; the actual Hub download is deferred to `kernelize` |

## Mapping semantics

**Keys (targets) come in two forms**, freely combinable with any value form:

- `type[nn.Module]` subclass: replace **every** instance whose exact type
  matches (`m.__class__ = impl_class`; subclasses are **not** touched)
- Dotted `str` path: `'pkg.mod.ClassName'` (resolves to an `nn.Module`
  subclass → same as a class key), `'pkg.mod.ClassName.forward'` (class method
  → `setattr`), `'pkg.mod.attr'` (module function → `setattr`). When a
  `transformers.*` family is not installed, the entry is silently skipped at
  DEBUG level (a missing family is the normal case)

**Values come in three forms**:

- A direct impl (class or function): no backend selection is performed; the
  generic installer installs it as-is. An impl class is **never `__init__`ed** —
  its forward must work against the attributes the original instance already has
- `KernelChoice(op='rms_norm', backends=('npu', 'liger'))`: pick the first
  available implementation in `backends` order — an unregistered backend, a
  failed `available()` check (with reason), or a load exception all fall through
  to the next backend; if none is available the original implementation is kept
- `HubRef`: a Hub reference built via `hub(...)`; loaded lazily

## Default config and customization

```python
model = kernelize(model)
# equivalent to
model = kernelize(model, DEFAULT_KERNEL_CONFIG)
```

`DEFAULT_KERNEL_CONFIG` covers: RMSNorm / RoPE / SwiGLU / MoE for Qwen2 / Qwen3
/ Qwen3-MoE / Qwen2.5-VL / Qwen3.5 / Qwen3.5-MoE (chain `('npu', 'liger')`,
CANN preferred on NPU); the 8 llama families and 4 gemma families (chain
`('liger',)`); plus the logical targets `'sdpa'` (global SDPA installation) and
`'fla'` (per-instance Qwen3.5 Flash Linear Attention patching).

**A user mapping fully replaces the default config — it is not merged.** Passing
only your own entries means no default entry is applied. To tweak on top of the
defaults, copy and merge:

```python
from twinkle.kernel import DEFAULT_KERNEL_CONFIG, KernelChoice, kernelize

model = kernelize(model, {**DEFAULT_KERNEL_CONFIG,
    # make rms_norm liger-first (falls back to npu if liger is unavailable)
    'transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm':
        KernelChoice(op='rms_norm', backends=('liger', 'npu')),
})
```

`DEFAULT_KERNEL_CONFIG` itself must not be mutated in place at runtime.

**Log levels**: with `mapping=None` (the default-config path), all
fallback/failure logs are DEBUG (a CUDA machine or a missing family is the
normal case); with an explicitly passed mapping (even a copy-merge of the
defaults) they are raised to WARNING — you stated an intent, so a no-op must be
reported. Every successful installation emits one INFO line:

```text
[kernelize] target=...Qwen3MLP.forward op=swiglu backend=npu installer=default
[kernelize] target=sdpa op=sdpa_attention backend=npu installer=install_sdpa
```

**CUDA behavior**: the default config is platform-uniform. On CUDA, the npu
backend's `available()` in a `('npu', 'liger')` chain returns False, so
selection falls through to liger; if liger is not installed the whole chain
fails (DEBUG skip under the default config ≈ no-op).

## Scenarios

### Pick an implementation by op name (with fallback order)

```python
model = kernelize(model, {**DEFAULT_KERNEL_CONFIG,
    'transformers.models.qwen3.modeling_qwen3.Qwen3MLP.forward':
        KernelChoice(op='swiglu', backends=('liger', 'npu')),   # liger first, npu as fallback
})
```

### String key + direct impl (bypass selection)

```python
from twinkle.kernel import kernelize
from twinkle.kernel.ops.rotary.npu import npu_apply_rotary_pos_emb

model = kernelize(model, {
    'transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb':
        npu_apply_rotary_pos_emb,
})
```

### Custom class replacement

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

model = kernelize(model, {Qwen2RMSNorm: MyRMSNorm})
```

### Hub kernel (HF Hub format)

```python
from twinkle.kernel import hub, kernelize

model = kernelize(model, {
    SiluAndMul: hub('kernels-community/activation:SiluAndMul', version=1),
})
```

Exactly one of `revision` or `version` is required. `hub(...)` triggers a lazy
import of the `kernels` package; if it is missing you will be told to
`pip install kernels`.

### GMM (grouped matmul) opt-in

The default config deliberately does **not** include the NPU replacement for
`transformers.integrations.moe._grouped_mm` (≈8x overhead without Expert
Parallelism). Add it explicitly when needed (note: a direct impl has no platform
gating — only do this on a confirmed NPU environment):

```python
from twinkle.kernel.ops.moe.npu import npu_grouped_mm

model = kernelize(model, {**DEFAULT_KERNEL_CONFIG,
    'transformers.integrations.moe._grouped_mm': npu_grouped_mm,
})
```

## The op registry (how to add an op / backend)

Built-in implementations are organized per operator under
`twinkle/kernel/ops/<op>/`: `__init__.py` performs registration (lazy references
and lightweight availability checks only — **it must not import optional
dependencies directly**), and `<backend>.py` holds the implementation itself.
Registering means declaring "backend X can provide an implementation for op Y":

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

- `KernelImpl.load(target)`: lazy loading factory, called only when the
  implementation is selected; receives the mapping target and may specialize per
  model family (e.g. liger's RMSNorm dispatches between gemma / qwen3_5 variants)
- `KernelImpl.available()`: returns `(True, None)` when usable, or
  `(False, reason)` to fall through; all platform and dependency checks live here
- Registering the same op name twice, or with empty implementations → `ValueError`
- **Vendor rule**: when an impl incorporates an external kernel, the file
  docstring must state the source repository@commit, the list of local changes,
  and a re-sync reminder

### Custom installers

Installations beyond plain class/attr replacement are provided by the op's own
installer (signature `(model, target, impl)`). Priority:
`KernelChoice.installer` → `OpDefinition.installer` → generic installer.
Example — SDPA writes into transformers' global attention registry:

```python
def install_sdpa(model, target, impl) -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    AttentionInterface._global_mapping['sdpa'] = impl
    ALL_ATTENTION_FUNCTIONS['sdpa'] = impl

register_op('sdpa_attention', implementations={...}, installer=install_sdpa)
```

The default config references such ops via logical targets
(`'sdpa': KernelChoice(op='sdpa_attention', backends=('npu',))`); a logical
target is only an identifier passed through to the installer and is not resolved
by the generic replacer. If an installer raises, the exception propagates — the
model must never be left in a half-installed state the user doesn't know about.

## Migration guide (old API → new)

| Old usage | New equivalent |
| --- | --- |
| `kernelize(model, npu_builtin(model))` | `kernelize(model)` (default config is CANN-first on NPU) |
| `kernelize(model, liger_builtin(model))` | Build an all-liger mapping, or `{**DEFAULT_KERNEL_CONFIG, ...}` with the targets you want rewritten to `('liger', ...)` chains |
| Manually merging `{**npu_builtin(m), **liger_builtin(m)}` | `{**DEFAULT_KERNEL_CONFIG, ...}` with per-target overrides |
| `{Qwen3RMSNorm: {'npu': NpuRMSNorm}}` (device-conditional dict) | `{Qwen3RMSNorm: NpuRMSNorm}` or `KernelChoice(op='rms_norm', backends=('npu',))` |
| `from twinkle.kernel.npu_impls.x import ...` | `from twinkle.kernel.ops.<op>.npu import ...` |
| `from twinkle.kernel.liger_impls.x import ...` | `from twinkle.kernel.ops.<op>.liger import ...` |

`npu_builtin()` / `liger_builtin()` / device-conditional dicts (`{'npu': impl}`)
have been removed with no compatibility shim; platform checks moved into
`KernelImpl.available()`. On NPU, `kernelize(model)`'s default replacement
result matches the old version; on CUDA it changes from a no-op to applying the
default config (effective wherever liger is installed, ≈ no-op otherwise).

## Environment variables

Only two remain:

- `TWINKLE_NPU_FLA`: Qwen3.5 FLA switch (on by default; set to `0`/`false` to disable)
- `TWINKLE_NPU_GATED_RMSNorm_FP32`: force Gated RMSNorm computation up to FP32 (off by default)

## Caveats

- `m.__class__ = impl_cls` is Python class-swap magic. An impl class **must**
  only override `forward` (plus helper methods) and must not define `__init__`,
  otherwise the original instance's attributes will mismatch the impl's
  expectations
- Matching is exact: `type(m) is target_cls`. Subclasses of `target_cls` are not
  replaced; if you need them replaced, list them in the mapping too
- Calling `kernelize` multiple times is idempotent (setting `__class__` to the
  impl again is harmless)
- There is no `unkernelize` — replacement is one-way
