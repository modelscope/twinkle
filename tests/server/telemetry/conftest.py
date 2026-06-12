# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared OTEL setup for telemetry tests.

OTel's global tracer provider is one-shot per process — the second
``trace.set_tracer_provider(...)`` call no-ops with a warning. So multiple
test modules that each tried to register their own provider would silently
share whichever one ran first, and tests that read spans from the wrong
exporter would fail. This fixture installs one provider + one in-memory
exporter for the entire telemetry test package.
"""
from __future__ import annotations

import pytest


def _otel_available() -> bool:
    try:
        from opentelemetry import trace  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture(scope='session')
def in_memory_span_exporter():
    if not _otel_available():
        pytest.skip('OTEL SDK not installed in test env')
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter
