# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end demo for ``twinkle.kernel.core.kernelize``.

Run with::

    conda run -n twinkle python scripts/kernelize_demo.py

The script exercises three replacement modes on CPU:

1. Class replacement  - rewrite ``__class__`` of matching ``nn.Module`` instances.
2. Attribute replacement - monkey-patch a module/function attribute via dotted path.
3. Hub replacement - lazy-load a kernel from a mocked ``kernels`` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the local ``src`` importable when running the script directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from twinkle.kernel.core import hub, kernelize


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _describe(obj) -> str:
    """Best-effort name for a callable (plain function or kernels ``Func``)."""
    qn = getattr(obj, '__qualname__', None)
    if qn:
        return qn
    return f"<{type(obj).__module__}.{type(obj).__name__}>"


class FusedQwen3MLP(Qwen3MLP):
    """Pretend fused kernel: same gated MLP + a constant +1.0 bias."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('[patched] FusedQwen3MLP.forward called')
        return super().forward(x) + 1.0


def _build_mlp() -> Qwen3MLP:
    config = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    return Qwen3MLP(config)


# --------------------------------------------------------------------------- #
# Demo 1: Class replacement
# --------------------------------------------------------------------------- #
def demo_class_replacement() -> None:
    print('=' * 60)
    print('Demo 1: class replacement (replace transformers Qwen3MLP)')
    print('=' * 60)

    mlp = _build_mlp()
    x = torch.randn(1, 4, mlp.config.hidden_size)

    out_before = mlp(x)
    print(f"Before kernelize: type = {type(mlp).__name__}")
    print(f"  output[0,0,:3] = {out_before[0, 0, :3].tolist()}")

    # Pass the MLP itself as the model; ``model.modules()`` yields it.
    kernelize(mlp, {Qwen3MLP: FusedQwen3MLP})

    out_after = mlp(x)
    print(f"After kernelize:  type = {type(mlp).__name__}")
    print(f"  output[0,0,:3] = {out_after[0, 0, :3].tolist()}")

    _assert(type(mlp) is FusedQwen3MLP, 'mlp should be FusedQwen3MLP after kernelize')
    # Params (gate_proj/up_proj/down_proj) are preserved on the instance, so the
    # only difference is the +1.0 added by the fused forward.
    _assert(
        torch.allclose(out_after, out_before + 1.0),
        'FusedQwen3MLP should add +1.0 to the original output',
    )
    print('✓ Class replacement passed\n')


# --------------------------------------------------------------------------- #
# Demo 2: Attribute replacement (patch transformers qwen3 apply_rotary_pos_emb)
# --------------------------------------------------------------------------- #
_QWEN3_MOD_PATH = 'transformers.models.qwen3.modeling_qwen3'
_ROPE_ATTR = 'apply_rotary_pos_emb'


def demo_attr_replacement() -> None:
    print('=' * 60)
    print('Demo 2: attribute replacement (two forms)')
    print('=' * 60)

    import importlib

    mod = importlib.import_module(_QWEN3_MOD_PATH)

    # ---- Form A: module attribute (pkg.mod.attr) -------------------------- #
    print('-' * 60)
    print('Form A: replace module-level function `apply_rotary_pos_emb`')
    print('-' * 60)

    original_rope = getattr(mod, _ROPE_ATTR)

    q = torch.ones(1, 2, 4, 8)
    k = torch.ones(1, 2, 4, 8)
    cos = torch.ones(1, 1, 4, 8)
    sin = torch.ones(1, 1, 4, 8)

    q_out_before, k_out_before = original_rope(q, k, cos, sin)
    print(f"Before kernelize: {_describe(mod.apply_rotary_pos_emb)}")

    def fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):  # noqa: ANN001
        return original_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    fused_apply_rotary_pos_emb._kernelize_marker = True  # type: ignore[attr-defined]

    try:
        kernelize(nn.Linear(1, 1), {f"{_QWEN3_MOD_PATH}.{_ROPE_ATTR}": fused_apply_rotary_pos_emb})

        patched_fn = getattr(mod, _ROPE_ATTR)
        print(f"After kernelize:  {_describe(patched_fn)}")
        _assert(patched_fn is fused_apply_rotary_pos_emb, 'module attr should be the fused fn')
        q_out_after, k_out_after = patched_fn(q, k, cos, sin)
        _assert(
            torch.allclose(q_out_after, q_out_before) and torch.allclose(k_out_after, k_out_before),
            'wrapped RoPE should preserve the original output',
        )
        print('✓ Form A (module attribute) passed\n')
    finally:
        setattr(mod, _ROPE_ATTR, original_rope)

    # ---- Form B: class attribute / method (pkg.mod.ClassName.attr) ------- #
    print('-' * 60)
    print('Form B: replace class method `Qwen3MLP.forward`')
    print('-' * 60)

    original_forward = Qwen3MLP.forward

    mlp = _build_mlp()
    x = torch.randn(1, 4, mlp.config.hidden_size)
    out_before = mlp(x)
    print(f"Before kernelize: {_describe(Qwen3MLP.forward)}")
    print(f"  output[0,0,:3] = {out_before[0, 0, :3].tolist()}")

    def fused_forward(self, x):  # noqa: ANN001
        return original_forward(self, x) + 1.0

    try:
        # Dotted path lands on the class, then setattr the method on it.
        kernelize(nn.Linear(1, 1), {f"{_QWEN3_MOD_PATH}.Qwen3MLP.forward": fused_forward})

        patched_forward = Qwen3MLP.forward
        print(f"After kernelize:  {_describe(patched_forward)}")
        _assert(patched_forward is fused_forward, 'class method should be the fused fn')

        out_after = mlp(x)
        print(f"  output[0,0,:3] = {out_after[0, 0, :3].tolist()}")
        _assert(
            torch.allclose(out_after, out_before + 1.0),
            'fused forward should add +1.0 to the original output',
        )
        print('✓ Form B (class method) passed\n')
    finally:
        setattr(Qwen3MLP, 'forward', original_forward)


# --------------------------------------------------------------------------- #
# Demo 3: Hub replacement (real HuggingFace Hub kernel via ``kernels``)
# --------------------------------------------------------------------------- #
# We use the real ``kernels-community/activation`` repo on the HF Hub, which
# ships a ``SiluAndMul`` layer (the SwiGLU activation used by Qwen3MLP).
#
# Note: the Hub kernel's ``forward`` calls a CUDA op, so it cannot *execute*
# on CPU. This demo verifies the parts that DO work on CPU: the kernel is
# downloaded lazily via ``_load_hub_ref`` and the target module's class is
# swapped to the Hub-loaded class. Running the fused forward requires CUDA.
_HUB_REPO = 'kernels-community/activation'
_HUB_LAYER = 'SiluAndMul'


class LocalSiluAndMul(nn.Module):
    """Pure-torch SwiGLU activation, same interface as the Hub ``SiluAndMul``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


def demo_hub_replacement() -> None:
    print('=' * 60)
    print('Demo 3: Hub replacement (real HF Hub kernel: kernels-community/activation)')
    print('=' * 60)

    try:
        from kernels import get_kernel  # noqa: F401
    except ImportError:
        print('Skipped: `kernels` package not installed (pip install kernels)')
        return

    model = nn.Sequential(LocalSiluAndMul())
    x = torch.randn(1, 4, 16, dtype=torch.float32)

    out_before = model(x)
    print(f"Before kernelize: type = {type(model[0]).__name__}")
    print(f"  output[0,0,:3] = {out_before[0, 0, :3].tolist()}")

    ref = hub(f"{_HUB_REPO}:{_HUB_LAYER}", version=1)
    print(f"HubRef: repo_id={ref.repo_id!r}, layer_name={ref.layer_name!r}, version={ref.version}")

    try:
        kernelize(model, {LocalSiluAndMul: ref})
    except Exception as e:
        print(f"Skipped: could not load Hub kernel ({type(e).__name__}: {e})")
        return

    hub_cls = type(model[0])
    print(f"After kernelize:  type = {hub_cls.__name__}")
    print(f"  module = {hub_cls.__module__}")

    _assert(hub_cls.__name__ == _HUB_LAYER, 'should be the Hub SiluAndMul class')
    _assert(
        'activation' in hub_cls.__module__,
        'loaded class should come from the Hub activation kernel package',
    )
    # The Hub forward is CUDA-only, so we do not execute it on CPU.
    print('(Hub kernel forward is CUDA-only; verified download + class swap on CPU)')
    print('✓ Hub replacement passed\n')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print('Running kernelize end-to-end demos on CPU...\n')
    demo_class_replacement()
    demo_attr_replacement()
    demo_hub_replacement()
    print('All demos passed.')


if __name__ == '__main__':
    main()
