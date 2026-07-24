# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-agnostic operator interfaces with per-platform implementations.

Each op family defines an abstract base class (e.g. ``EpExpertsGmm``) plus a
dispatcher that picks the first eligible backend implementation from a
lazily-built registry. Callers invoke the dispatcher only; backend details
(NPU, GPU, ...) stay hidden behind the interface.
"""
from .ep import EpExpertsGmm, ep_forward

__all__ = [
    'EpExpertsGmm',
    'ep_forward',
]
