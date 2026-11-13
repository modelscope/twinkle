# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fused lm_head projection and selected-token log-probability computation."""

from .functional import fused_linear_selective_log_softmax

__all__ = ['fused_linear_selective_log_softmax']
