# Copyright (c) ModelScope Contributors. All rights reserved.
"""Correlation key constants for business-layer spans.

All correlation attributes share the ``twinkle.`` prefix so operators can
filter Tempo / Loki by ``twinkle.session_id``, ``twinkle.model_id``, etc.
The ``set_correlation_attrs`` helper attaches only present (non-None) values
so partially-known operations don't end up with empty attributes.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PREFIX = 'twinkle.'

SESSION_ID = f'{PREFIX}session_id'
MODEL_ID = f'{PREFIX}model_id'
REPLICA_ID = f'{PREFIX}replica_id'
TOKEN_ID = f'{PREFIX}token_id'
SAMPLING_SESSION_ID = f'{PREFIX}sampling_session_id'
BASE_MODEL = f'{PREFIX}base_model'

CORRELATION_KEYS: tuple[str, ...] = (
    SESSION_ID,
    MODEL_ID,
    REPLICA_ID,
    TOKEN_ID,
    SAMPLING_SESSION_ID,
    BASE_MODEL,
)


def set_correlation_attrs(span: Any, values: Mapping[str, Any] | None) -> None:
    """Attach the given correlation attributes to ``span``.

    Skips ``None`` values and is a no-op for NoOp spans returned when the
    OTEL SDK is not installed.
    """
    if not values or span is None:
        return
    setter = getattr(span, 'set_attribute', None)
    if setter is None:
        return
    for key, value in values.items():
        if value is None:
            continue
        setter(key, value)
