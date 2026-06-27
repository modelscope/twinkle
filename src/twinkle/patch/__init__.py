# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from contextlib import contextmanager
from typing import Any, Type, Union

from .base import Patch


def _resolve(patch_cls: Union[Patch, Type[Patch], str]) -> Patch:
    from twinkle.utils import construct_class
    return construct_class(patch_cls, Patch, sys.modules[__name__])


def apply_patch(module: Any, patch_cls: Union[Patch, Type[Patch], str], *args, **kwargs):
    patch_ins = _resolve(patch_cls)
    return patch_ins(module, *args, **kwargs)


@contextmanager
def apply_context(module: Any, patch_cls: Union[Patch, Type[Patch], str], *args, **kwargs):
    # Apply patch on enter; revert via subclass-implemented unpatch on exit (even on exception).
    patch_ins = _resolve(patch_cls)
    result = patch_ins(module, *args, **kwargs)
    try:
        yield result
    finally:
        patch_ins.unpatch(module, *args, **kwargs)


__all__ = ['apply_patch', 'apply_context', 'Patch']
