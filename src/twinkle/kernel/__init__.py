# Copyright (c) ModelScope Contributors. All rights reserved.
"""Mapping-driven kernel replacement.

Public symbols:

- :func:`kernelize`              apply ``mapping`` to a model
- :func:`hub`                    build a Hub kernel reference
- :class:`KernelChoice`          per-target op + backend-priority selection
- :data:`DEFAULT_KERNEL_CONFIG`  the built-in default mapping (copy/merge to customize)
"""
from . import ops  # noqa: F401  triggers built-in op registration (must happen before the first kernelize() call)
from .config import DEFAULT_KERNEL_CONFIG
from .core import hub, kernelize
from .registry import KernelChoice

__all__ = ['kernelize', 'hub', 'KernelChoice', 'DEFAULT_KERNEL_CONFIG']
