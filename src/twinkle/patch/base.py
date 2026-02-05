# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Union, Type
from twinkle.utils import construct_class


class Patch:

    def patch(self, module, *args, **kwargs):
        ...
