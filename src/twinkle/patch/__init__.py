# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import Any, Union, Type

from .base import Patch


def apply_patch(module: Any, patch_cls: Union[Patch, Type[Patch], str], *args, **kwargs):
    from ..utils import construct_class
    patch_ins = construct_class(patch_cls, Patch, sys.modules[__name__])
    return patch_ins(module, *args, **kwargs)

__all__ = ['apply_patch', 'Patch']
