# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Any, List, Union

if TYPE_CHECKING:
    import torch


class Patch:

    def __call__(self, module: Union['torch.nn.Module', List['torch.nn.Module'], Any], *args, **kwargs):
        ...
