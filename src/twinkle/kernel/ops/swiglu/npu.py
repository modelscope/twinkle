# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fused SwiGLU forward for Ascend NPU."""
from __future__ import annotations

import torch


def npu_swiglu_forward(self, hidden_state):
    """Fused Qwen-style SwiGLU.

    Used as a class-attribute replacement on HF MLP classes.
    Required instance attributes: ``gate_proj``, ``up_proj``, ``down_proj``.
    """
    import torch_npu
    return self.down_proj(
        torch_npu.npu_swiglu(
            torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1),
            dim=-1,
        ))
