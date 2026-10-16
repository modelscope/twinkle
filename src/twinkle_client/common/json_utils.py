# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lightweight JSON conversion helpers shared by component clients and servers."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from typing import Any

from pydantic import BaseModel


_PRIMITIVE_TYPES = (str, Number, bool, bytes, type(None))


def json_safe(obj: Any) -> Any:
    """Recursively convert models, tensors, and arrays into JSON-compatible values.

    This module intentionally does not import :mod:`twinkle`: component protocol
    types are imported while the top-level package is still being initialized.
    """
    if isinstance(obj, BaseModel):
        return json_safe(obj.model_dump())
    if isinstance(obj, Mapping):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_safe(value) for value in obj]
    tolist = getattr(obj, 'tolist', None)
    if callable(tolist) and not isinstance(obj, _PRIMITIVE_TYPES):
        try:
            return json_safe(tolist())
        except Exception:
            pass
    return obj
