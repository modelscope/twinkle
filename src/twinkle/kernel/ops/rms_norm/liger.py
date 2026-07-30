# Copyright (c) ModelScope Contributors. All rights reserved.
"""Class-replacement RMSNorm adapters for Liger Kernel.

Designed for ``kernelize`` class replacement: do not define ``__init__``; rely
on the attributes already present on the original HuggingFace instance
(``weight``, ``variance_epsilon`` *or* ``eps``). The Liger-specific attributes
(``offset`` / ``casting_mode`` / ``in_place`` / ``row_mode``) are set lazily on
the first ``forward`` with family-appropriate defaults, mirroring Liger's own
``_patch_rms_norm_module`` (monkey_patch.py) so no global state is mutated.

Per-family defaults (matching ``liger_kernel.transformers.monkey_patch``):

  - llama / qwen / phi3 / olmo2 / glm4 / granite / internvl:
        offset=0.0, casting_mode="llama", in_place=True
  - gemma / gemma2 / gemma3_text / paligemma:
        offset=1.0, casting_mode="gemma", in_place=False
  - gemma4:
        offset=0.0, casting_mode="gemma", in_place=False
  - qwen3_5 / qwen3_5_moe:
        offset=1.0, casting_mode="gemma", in_place=False
        (residual parameterization: scale = 1 + weight, weight init=0)
"""
from __future__ import annotations

import torch.nn as nn
from liger_kernel.ops import LigerRMSNormFunction


class _LigerRMSNormBase(nn.Module):
    """Base adapter; subclasses set the family-specific defaults below."""

    _liger_offset: float = 0.0
    _liger_casting_mode: str = 'llama'
    _liger_in_place: bool = True
    _liger_row_mode = None

    def _ensure_liger_attrs(self) -> None:
        if getattr(self, '_liger_attrs_set', False):
            return
        # ``variance_epsilon`` is the Liger name; HF names it either
        # ``variance_epsilon`` (newer) or ``eps`` (older). ``_patch_rms_norm_module``
        # resolves the same way.
        if not hasattr(self, 'variance_epsilon'):
            self.variance_epsilon = getattr(self, 'eps', 1e-6)
        self.offset = self._liger_offset
        self.casting_mode = self._liger_casting_mode
        self.in_place = self._liger_in_place
        self.row_mode = self._liger_row_mode
        self._liger_attrs_set = True

    def forward(self, hidden_states):
        self._ensure_liger_attrs()
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
            self.row_mode,
        )

    def extra_repr(self):
        self._ensure_liger_attrs()
        return (f'weight_shape={tuple(self.weight.shape) if self.weight is not None else None}, '
                f'eps={self.variance_epsilon}, offset={self.offset}, '
                f'in_place={self.in_place}, row_mode={self.row_mode}')


class LigerRMSNormReplacement(_LigerRMSNormBase):
    """llama / qwen / phi3 family: offset=0.0, casting_mode='llama', in_place=True."""

    _liger_offset = 0.0
    _liger_casting_mode = 'llama'
    _liger_in_place = True


class LigerRMSNormGemmaReplacement(_LigerRMSNormBase):
    """gemma / gemma2 / gemma3_text: offset=1.0, casting_mode='gemma', in_place=False."""

    _liger_offset = 1.0
    _liger_casting_mode = 'gemma'
    _liger_in_place = False


class LigerRMSNormGemma4Replacement(_LigerRMSNormBase):
    """gemma4: offset=0.0, casting_mode='gemma', in_place=False.

    Note: Gemma4RMSNorm has a ``with_scale=False`` variant (no weight, used for
    attention ``v_norm``). That path falls back to a plain torch RMSNorm; this
    adapter is only wired onto the scale-bearing variants — see ``liger_builtin``.
    """

    _liger_offset = 0.0
    _liger_casting_mode = 'gemma'
    _liger_in_place = False


class LigerRMSNormQwen35Replacement(_LigerRMSNormBase):
    """qwen3_5 / qwen3_5_moe: offset=1.0, casting_mode='gemma', in_place=False.

    Qwen3.5 uses residual parameterization (``weight`` init=0, scale = 1 + weight),
    matching the gemma casting-mode kernel which computes
    ``norm(x) * (offset + weight)`` = ``(1 + weight) * norm(x)`` with offset=1.0.
    The gemma precision path (fp32 throughout, cast back at end) also matches
    HF ``Qwen3_5RMSNorm.forward`` which does ``_norm(x.float()) * (1 + w.float())``.

    Using ``LigerRMSNormReplacement`` (llama cast, offset=0) here produces
    ``weight * norm(x)`` ≈ 0 (since weight≈0), causing NaN in attention and loss.
    """

    _liger_offset = 1.0
    _liger_casting_mode = 'gemma'
    _liger_in_place = False
