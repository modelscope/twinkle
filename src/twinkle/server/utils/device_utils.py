# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from collections.abc import MutableMapping
from functools import wraps
from typing import Any, Callable


def auto_fill_device_group_visible_devices(kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Fill `device_group.visible_devices` from env for server app builders."""
    auto_patch = os.environ.get('TWINKLE_AUTO_VISIBLE_DEVICES_FROM_ENV', '1')
    if str(auto_patch).lower() in {'0', 'false', 'no', 'off'}:
        return kwargs

    device_group = kwargs.get('device_group')
    if not isinstance(device_group, MutableMapping):
        return kwargs
    if device_group.get('visible_devices'):
        return kwargs

    visible_devices = os.environ.get('ASCEND_RT_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible_devices:
        return kwargs

    patched = dict(kwargs)
    patched_group = dict(device_group)
    patched_group['visible_devices'] = visible_devices
    patched['device_group'] = patched_group
    return patched


def wrap_builder_with_device_group_env(builder: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap app builder and auto-fill device_group.visible_devices from env."""

    @wraps(builder)
    def _wrapped(*args, **kwargs):
        return builder(*args, **auto_fill_device_group_visible_devices(kwargs))

    return _wrapped
