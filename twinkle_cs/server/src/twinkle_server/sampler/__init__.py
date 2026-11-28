# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler deployment package.

``build_sampler_app`` is exposed lazily via ``__getattr__`` so importing the
package doesn't eagerly pull in ``app.py`` and its transitive dependencies.
"""
from __future__ import annotations

__all__ = ['build_sampler_app']


def __getattr__(name: str):
    if name == 'build_sampler_app':
        from .app import build_sampler_app

        return build_sampler_app
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
