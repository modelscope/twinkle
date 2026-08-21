# Copyright (c) ModelScope Contributors. All rights reserved.
"""GeGLU forward-replacement for Liger Kernel.

Used as a *function-level* mapping value (string key ``'<module>.<Cls>.forward'``).
Reads only ``gate_proj`` / ``up_proj`` / ``down_proj`` — the same attributes
HuggingFace GeGLU MLP variants define — so it is a safe function-level
mapping value.
"""
from __future__ import annotations

from liger_kernel.ops import LigerGELUMulFunction


def liger_geglu_forward(self, x):
    """GeGLU forward replacement for the gemma family (gemma / gemma2 / gemma3 / gemma4).

    Uses the tanh GELU approximation, matching Liger's own ``LigerGEGLUMLP``
    and HF's gemma activation choice.
    """
    return self.down_proj(LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
