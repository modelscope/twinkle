# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from collections.abc import Mapping
from typing import Callable


def any_callable(args):
    if isinstance(args, Mapping):
        return any(any_callable(arg) for arg in args.values())
    elif isinstance(args, (tuple, list, set)):
        return any(any_callable(arg) for arg in args)
    else:
        return isinstance(args, (Callable, type))


def check_unsafe(*args, **kwargs):
    if not trust_remote_code():
        if any_callable(args) or any_callable(kwargs):
            raise ValueError(
                "Twinkle does not support Callable or Type inputs in safe mode."
            )


def trust_remote_code():
    return os.environ.get("TRUST_REMOTE_CODE", "0") != "0"
