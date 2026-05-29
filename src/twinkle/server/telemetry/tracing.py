"""Twinkle Server tracing utilities — thin wrapper over OpenTelemetry tracing."""

from __future__ import annotations

try:
    from opentelemetry import trace
    from opentelemetry.propagate import inject, extract
    from opentelemetry.context import Context
    _OTEL_AVAILABLE = True
except Exception:
    _OTEL_AVAILABLE = False


def get_tracer(name: str = "twinkle-server"):
    """Retrieve tracer instance. Returns NoOp tracer when OTEL is not installed."""
    if not _OTEL_AVAILABLE:
        return _NoopTracer()
    return trace.get_tracer(name)


def inject_context(carrier: dict) -> None:
    """Inject current trace context into carrier. Noop when OTEL is not installed."""
    if not _OTEL_AVAILABLE:
        return
    inject(carrier)


def extract_context(carrier: dict):
    """Extract trace context from carrier. Returns empty context when OTEL is not installed."""
    if not _OTEL_AVAILABLE:
        return None
    return extract(carrier)


def get_current_span():
    """Get current active span. Returns noop span when OTEL is not installed."""
    if not _OTEL_AVAILABLE:
        return _NoopSpan()
    return trace.get_current_span()


class _NoopSpan:
    """Minimal noop span for when OTEL is not available."""
    def set_attribute(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass
    def add_event(self, *args, **kwargs): pass
    def end(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class _NoopTracer:
    """Minimal noop tracer for when OTEL is not available."""
    def start_as_current_span(self, name, **kwargs):
        return _NoopSpan()
    def start_span(self, name, **kwargs):
        return _NoopSpan()
