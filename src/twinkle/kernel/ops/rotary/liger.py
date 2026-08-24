# Copyright (c) ModelScope Contributors. All rights reserved.
"""Liger RoPE adapter: re-export liger_kernel's rotary implementation.

Loaded lazily via ``registry.lazy_import`` — importing this module pulls in
``liger_kernel``, so it must only be imported after ``is_liger_available()``
has passed. Compatible with full-rotation families only; partial-RoPE
families (qwen3_5) are excluded by the per-target backend chains in
``config.py``.
"""
from __future__ import annotations

from liger_kernel.transformers import liger_rotary_pos_emb

__all__ = ['liger_rotary_pos_emb']
