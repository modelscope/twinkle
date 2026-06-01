# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler deployment package.

``build_sampler_app`` is exposed lazily via ``__getattr__`` so importing the
mock backend (``twinkle.server.sampler.backends.mock_sampler``) on a CPU-only
host doesn't pull in vllm via ``app.py`` at package-init time (R2.2, R4.3).
"""
from __future__ import annotations

__all__ = ['build_sampler_app']


def __getattr__(name: str):
    if name == 'build_sampler_app':
        from .app import build_sampler_app

        return build_sampler_app
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
