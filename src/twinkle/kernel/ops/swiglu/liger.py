# Copyright (c) ModelScope Contributors. All rights reserved.
"""SwiGLU forward-replacement for Liger Kernel.

Used as a *function-level* mapping value (string key ``'<module>.<Cls>.forward'``)
so it composes with the existing SwiGLU forward-replacement pattern. Reads only
``gate_proj`` / ``up_proj`` / ``down_proj``, which every HuggingFace SwiGLU MLP
variant (Qwen2MLP, Qwen3MLP, LlamaMLP, MistralMLP, ...) already defines, so no
``__init__`` and no per-instance attribute setup is required.

For Qwen3-MoE the expert/MLP classes differ; class replacement with Liger's
own ``LigerExperts`` is wired in ``ops/moe/liger.py``.
"""
from __future__ import annotations

from liger_kernel.ops import LigerSiLUMulFunction


def liger_swiglu_forward(self, x):
    return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
