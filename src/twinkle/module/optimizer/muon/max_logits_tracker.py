# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import torch.nn.functional as F
from contextlib import suppress
from typing import Optional

from twinkle.utils import get_logger

logger = get_logger()


class MaxLogitsTracker:
    """The largest attention logit seen since the last read, for QK-Clip to scale against.

    QK-Clip needs a quantity the training loop never produces: the peak pre-softmax attention score.
    Attention implementations do not report it, and there is no argument to ask for it, so it is taken
    by wrapping the functions that compute it. The wrapping is global and permanent once installed,
    which is why :class:`MuonClip` only installs it when QK-Clip is actually on.

    What gets recorded depends on the attention implementation:

    - Eager attention reaches ``softmax`` with the scores themselves, so the value is exact.
    - SDPA and FlashAttention keep the scores internal, so only a bound is available:
      ``max(qk^T * scale) <= max||q|| * max||k|| * scale``. QK-Clip then triggers earlier than it
      strictly needs to, which errs toward clipping rather than toward missing a spike.

    One scalar covers the whole step -- not per layer, not per head -- because that is what the clip
    decision uses.

    The maximum is kept as a device tensor rather than a Python float: reading it per call would
    synchronise the device on every attention layer of every step, and the only consumer is
    ``step()``, once.
    """

    _max_logits: Optional[torch.Tensor] = None
    _installed = False

    _orig_torch_softmax = None
    _orig_F_softmax = None
    _orig_sdpa = None
    _orig_flash_attn_func = None

    @classmethod
    def _update(cls, value: torch.Tensor) -> None:
        value = value.detach().float().reshape(())
        if cls._max_logits is None:
            cls._max_logits = value
        else:
            cls._max_logits = torch.maximum(cls._max_logits, value.to(cls._max_logits.device))

    @classmethod
    def consume(cls) -> Optional[torch.Tensor]:
        """The maximum since the last call, and start over. ``None`` if nothing was recorded."""
        value, cls._max_logits = cls._max_logits, None
        return value

    @classmethod
    def install(cls) -> None:
        """Wrap the attention entry points. Idempotent; there is no matching uninstall."""
        if cls._installed:
            return
        cls._installed = True
        cls._install_softmax()
        cls._install_sdpa()
        cls._install_flash_attn()

    @classmethod
    def _install_softmax(cls) -> None:
        cls._orig_torch_softmax = torch.softmax
        cls._orig_F_softmax = F.softmax

        def capture(x, dim):
            # Attention scores are 4-D ([B, H, Lq, Lk]) and reduced along the last axis. Anything else
            # is some other softmax in the model and must not be mistaken for an attention score.
            if not isinstance(x, torch.Tensor) or x.dim() != 4:
                return
            if dim is None or dim not in (-1, x.dim() - 1):
                return
            with suppress(Exception):
                cls._update(x.amax())

        def torch_softmax(x, dim=None, dtype=None):
            with suppress(Exception):
                capture(x, dim)
            return cls._orig_torch_softmax(x, dim=dim, dtype=dtype)

        def f_softmax(x, dim=None, _stacklevel=3, dtype=None):
            with suppress(Exception):
                capture(x, dim)
            return cls._orig_F_softmax(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        torch.softmax = torch_softmax
        F.softmax = f_softmax

    @classmethod
    def _install_sdpa(cls) -> None:
        if not hasattr(F, 'scaled_dot_product_attention'):
            return
        cls._orig_sdpa = F.scaled_dot_product_attention

        def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            with suppress(Exception):
                cls._update(cls._logit_bound(query, key, scale))
            return cls._orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa)

        F.scaled_dot_product_attention = sdpa

    @classmethod
    def _install_flash_attn(cls) -> None:
        try:
            import flash_attn.flash_attn_interface as flash_interface
        except Exception:  # noqa: BLE001  -- flash-attn is optional
            return
        cls._orig_flash_attn_func = flash_interface.flash_attn_func

        def flash_attn(q,
                       k,
                       v,
                       dropout_p=0.0,
                       softmax_scale=None,
                       causal=False,
                       window_size=(-1, -1),
                       alibi_slopes=None,
                       deterministic=False,
                       return_attn_probs=False):
            with suppress(Exception):
                cls._update(cls._logit_bound(q, k, softmax_scale))
            return cls._orig_flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs)

        flash_interface.flash_attn_func = flash_attn

    @staticmethod
    def _logit_bound(query: torch.Tensor, key: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
        """``max||q|| * max||k|| * scale``: what a fused attention allows knowing about its logits."""
        q_norm = query.detach().float().norm(p=2, dim=-1).amax()
        k_norm = key.detach().float().norm(p=2, dim=-1).amax()
        if scale is None:
            scale = 1.0 / math.sqrt(float(query.size(-1)))
        return q_norm * k_norm * float(scale)
