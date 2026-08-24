# Copyright (c) ModelScope Contributors. All rights reserved.
"""Grouped-matmul interface for expert-parallel (EP) experts compute.

``EpExpertsGmm`` is the abstract backend interface; each platform provides a
subclass (e.g. ``NpuEpExpertsGmm`` in ``ep_experts_gmm_npu.py``) that is
registered lazily. The generic per-expert loop (``LoopEpExpertsGmm`` in
``ep_experts_gmm_loop.py``) is registered last as the always-eligible
fallback. ``ep_forward`` is the single entry point used by the EP
forward: it dispatches to the first eligible backend, so it always returns a
result.
"""
from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from torch import nn

from twinkle import get_logger

logger = get_logger()


class EpExpertsGmm(ABC):
    """Backend interface for batched EP experts compute via grouped matmul.

    A backend computes all local experts in one shot from the permuted token
    buffer (tokens sorted by expert, contiguous chunks per local expert),
    replacing the per-expert Python loop.
    """

    name: str
    # Fallback backends (e.g. the per-expert loop) are always eligible and are
    # registered last; hitting one means every accelerated backend declined.
    fallback: bool = False

    @abstractmethod
    def ineligible_reason(self, experts_mod: nn.Module) -> str | None:
        """Return why this backend cannot handle ``experts_mod``, or None if it can."""

    @abstractmethod
    def forward(
        self,
        experts_mod: nn.Module,
        permuted_tokens: torch.Tensor,
        num_global_sum_tokens_per_local_expert: torch.Tensor,
        experts_per_rank: int,
    ) -> torch.Tensor:
        """Compute local experts on the permuted token buffer."""


_IMPLS: list[EpExpertsGmm] | None = None
_PATH_LOGGED = False
_WARN_LOGGED = False


def _get_impls() -> list[EpExpertsGmm]:
    """Build the backend registry on first use.

    Each backend module is imported defensively: platforms lacking its
    dependencies (e.g. no torch_npu) simply skip that backend.
    """
    global _IMPLS
    if _IMPLS is None:
        _IMPLS = []
        try:
            from .npu import NpuEpExpertsGmm
            _IMPLS.append(NpuEpExpertsGmm())
        except ImportError:
            pass
        # The loop fallback has no optional dependencies and is always last:
        # it guarantees the dispatcher never runs out of backends.
        from .loop import LoopEpExpertsGmm
        _IMPLS.append(LoopEpExpertsGmm())
    return _IMPLS


def ep_forward(
    experts_mod: nn.Module,
    permuted_tokens: torch.Tensor,
    num_global_sum_tokens_per_local_expert: torch.Tensor,
    experts_per_rank: int,
) -> torch.Tensor:
    """Dispatch to the first eligible backend (the loop fallback guarantees a hit).

    One-time logging: INFO on the first accelerated dispatch (process-wide);
    WARNING with the ineligibility reasons when falling back to the loop
    (once per experts instance, as reasons may differ per MoE block).
    """
    if permuted_tokens.numel() == 0:
        # Preserve the autograd edge to token_pre_all2all. Returning a new
        # empty tensor can make this rank skip the matching backward
        # all-to-all, causing EP collective order divergence.
        return permuted_tokens

    global _PATH_LOGGED, _WARN_LOGGED
    ineligible: list[str] = []
    for impl in _get_impls():
        reason = impl.ineligible_reason(experts_mod)
        if reason is not None:
            ineligible.append(f'{impl.name}: {reason}')
            continue
        if impl.fallback:
            if ineligible and not _WARN_LOGGED:
                detail = '; '.join(ineligible)
                logger.warning(f'EP experts compute: grouped matmul disabled ({detail}); '
                               f'falling back to {impl.name}.')
                _WARN_LOGGED = True
        elif not _PATH_LOGGED:
            logger.info(f'EP experts compute: using {impl.name} grouped matmul (per-expert loop replaced).')
            _PATH_LOGGED = True
        return impl.forward(experts_mod, permuted_tokens, num_global_sum_tokens_per_local_expert, experts_per_rank)
    raise RuntimeError('no EP experts backend registered (loop fallback missing)')
