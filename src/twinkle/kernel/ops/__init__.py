# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-agnostic operator interfaces with per-platform implementations.

Each op family defines an abstract base class (e.g. ``EpExpertsGmm``) plus a
dispatcher that picks the first eligible backend implementation from a
lazily-built registry. Callers invoke the dispatcher only; backend details
(NPU, GPU, ...) stay hidden behind the interface.

Importing this package also registers all built-in kernel ops (swiglu,
rms_norm, rotary, geglu, moe, sdpa_attention, fla) into
``twinkle.kernel.registry`` — registration modules are lightweight (lazy
references + availability checks only, no optional-dependency imports).
"""
# Trigger built-in op registration (must happen before the first kernelize() call)
from . import fla, geglu, moe, rms_norm, rotary, sdpa_attention, swiglu  # noqa: F401,E402
from .ep import EpExpertsGmm, ep_forward

__all__ = [
    'EpExpertsGmm',
    'ep_forward',
]
