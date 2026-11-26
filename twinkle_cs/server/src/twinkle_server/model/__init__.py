# Copyright (c) ModelScope Contributors. All rights reserved.
"""Model deployment package.

``build_model_app`` is exposed lazily via ``__getattr__`` so importing the
package doesn't eagerly pull in ``app.py`` and its transitive dependencies.
"""
from __future__ import annotations

__all__ = ['build_model_app']


def __getattr__(name: str):
    if name == 'build_model_app':
        from .app import build_model_app

        return build_model_app
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
