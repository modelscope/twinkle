# Copyright (c) ModelScope Contributors. All rights reserved.
"""Liger MoE experts adapter: class replacement via ``LigerExperts``.

Only effective when the user explicitly puts ``'liger'`` in the
``moe_experts`` backend chain — the default chain is ``('npu',)`` so the
CANN grouped-matmul forward-level replacement is never shadowed.
"""
from __future__ import annotations

from liger_kernel.transformers import LigerExperts

__all__ = ['LigerExperts']
