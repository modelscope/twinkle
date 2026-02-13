# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Type, Union

from twinkle.utils import construct_class


class Patch:

    def __call__(self, module, *args, **kwargs):
        ...
