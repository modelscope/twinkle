# Copyright (c) ModelScope Contributors. All rights reserved.
"""Self-compiled ACLNN C++ extensions for Ascend NPU fusion operators.

Provides JIT-compiled bindings for DeepSeek-V4 SAS (Sparse Attention with
Shared-KV) and LI (Lightning Indexer) without depending on mindspeed.
"""
from .builder import build_op

__all__ = ['build_op']
