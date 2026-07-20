# Copyright (c) ModelScope Contributors. All rights reserved.
"""SwiGLU forward-replacement for Liger Kernel.

Used as a *function-level* mapping value (string key ``'<module>.<Cls>.forward'``)
so it composes with the existing SwiGLU forward-replacement pattern in
``builtin.py`` (``_add_swiglu_if_present``). Reads only ``gate_proj`` /
``up_proj`` / ``down_proj``, which every HuggingFace SwiGLU MLP variant
(Qwen2MLP, Qwen3MLP, LlamaMLP, MistralMLP, ...) already defines, so no
``__init__`` and no per-instance attribute setup is required.

For Qwen3-MoE the expert/MLP classes differ; class replacement with Liger's
own ``LigerQwen3MoeSwiGLUMLP`` / ``LigerExperts`` is wired directly in
``liger_builtin`` since those read matching attributes.
"""
from __future__ import annotations

from liger_kernel.ops import LigerGELUMulFunction, LigerSiLUMulFunction


def liger_swiglu_forward(self, x):
    return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


def liger_geglu_forward(self, x):
    """GeGLU forward replacement for the gemma family (gemma / gemma2 / gemma3 / gemma4).

    Reads only ``gate_proj`` / ``up_proj`` / ``down_proj`` — the same attributes
    HuggingFace GeGLU MLP variants define — so it is a safe function-level
    mapping value. Uses the tanh GELU approximation, matching Liger's own
    ``LigerGEGLUMLP`` and HF's gemma activation choice.
    """
    return self.down_proj(LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
