# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trace context carrier helpers for cross-deployment propagation (R13).

``DeploymentHandle`` calls between Ray Serve deployments do not go through
HTTP, so the existing ``inject_context(headers)`` path doesn't apply. To keep
a single trace continuous across an internal handle hop, callers serialize
the active OpenTelemetry context into a small dict using :func:`make_carrier`
and pass it as a kwarg; the receiving side calls :func:`activate_carrier`
inside its handler so subsequent spans attach as children of the propagated
context.

When the OTEL SDK is missing, both helpers degrade to a NoOp: ``make_carrier``
returns an empty dict and ``activate_carrier`` is a no-op context manager
that runs the body without raising (R13.4 / R18.3).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

try:
    from opentelemetry import context as _otel_context  # type: ignore
    from opentelemetry.propagate import extract as _otel_extract, inject as _otel_inject  # type: ignore

    _OTEL_AVAILABLE = True
except Exception:
    _OTEL_AVAILABLE = False


def make_carrier() -> dict[str, Any]:
    """Inject the active trace context into a fresh dict (R13.1).

    Returns an empty dict when OTEL is not installed; the receiving side's
    :func:`activate_carrier` handles that case gracefully.
    """
    carrier: dict[str, Any] = {}
    if not _OTEL_AVAILABLE:
        return carrier
    try:
        _otel_inject(carrier)
    except Exception:
        # NEVER let a tracing failure block the underlying RPC.
        return {}
    return carrier


@contextmanager
def activate_carrier(carrier: Mapping[str, Any] | None) -> Iterator[None]:
    """Attach the trace context from ``carrier`` for the lifetime of the block.

    Subsequent spans started inside the block become children of the
    propagated context (R13.2). When ``carrier`` is ``None`` or empty, or
    when OTEL is not installed, the block runs without attaching any
    context and a fresh trace is started (R13.4).
    """
    if not _OTEL_AVAILABLE or not carrier:
        yield
        return

    try:
        ctx = _otel_extract(dict(carrier))
    except Exception:
        yield
        return

    token = None
    try:
        token = _otel_context.attach(ctx)
        yield
    finally:
        if token is not None:
            try:
                _otel_context.detach(token)
            except Exception:
                pass
