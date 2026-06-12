"""Twinkle Server tracing utilities — thin wrapper over OpenTelemetry tracing."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from fastapi import Request
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.propagate import extract, inject
    _OTEL_AVAILABLE = True
except Exception:
    _OTEL_AVAILABLE = False

from .correlation import set_correlation_attrs


def get_tracer(name: str = 'twinkle-server'):
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

    def set_attribute(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def add_event(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoopTracer:
    """Minimal noop tracer for when OTEL is not available."""

    def start_as_current_span(self, name, **kwargs):
        return _NoopSpan()

    def start_span(self, name, **kwargs):
        return _NoopSpan()


@contextmanager
def traced_operation(
    name: str,
    *,
    attrs: Mapping[str, Any] | None = None,
    tracer_name: str = 'twinkle.server.business',
) -> Iterator[Any]:
    """Run a business-layer block under one OTEL span.

    The span starts before the block runs and ends after it returns. If the
    block raises, the exception is recorded on the span, the span status is
    set to ERROR, the span is ended, and the original exception is re-raised
    to the caller. When the OTEL SDK is missing, the context manager degrades
    to a NoOp that runs the block normally and returns the same result it
    would return when tracing is active.
    """
    if not _OTEL_AVAILABLE:
        yield _NoopSpan()
        return

    tracer = trace.get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        if attrs:
            set_correlation_attrs(span, attrs)
        try:
            yield span
        except Exception as exc:
            try:
                span.record_exception(exc)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            except Exception:
                # NEVER let a tracing error mask the underlying exception.
                pass
            raise


def create_tracing_middleware(service_component: str):
    """Create an HTTP tracing middleware compatible with Ray Serve pickling.

    Unlike ``FastAPIInstrumentor.instrument_app`` which attaches unpicklable
    references (e.g. ``_thread.lock``) to the FastAPI app and breaks Ray Serve
    deployment pickling, the returned middleware is a plain async function
    closing only over the ``service_component`` string.

    When OpenTelemetry is not installed, returns a passthrough middleware so
    the server still works without the optional dependency.

    Args:
        service_component: Logical service name used as the tracer name suffix
            and recorded as a span attribute (e.g. ``Gateway``, ``Model``,
            ``Processor``, ``Sampler``).

    Returns:
        An async FastAPI HTTP middleware function.
    """
    if not _OTEL_AVAILABLE:

        async def passthrough_middleware(request: Request, call_next):
            return await call_next(request)

        return passthrough_middleware

    async def tracing_middleware(request: Request, call_next):
        tracer = trace.get_tracer(f'twinkle.server.{service_component}')

        method = request.method
        path = request.url.path
        span_name = f'{method} {path}'

        # Continue an upstream trace when the inbound request carries trace
        # headers (e.g. a Gateway -> Model / Gateway -> Sampler hop), so the
        # SERVER span attaches to the propagated context instead of starting a
        # fresh, disconnected trace. A request with no trace headers yields an
        # empty context and a new trace is started normally.
        ctx = extract(request.headers)

        with tracer.start_as_current_span(
                span_name,
                kind=trace.SpanKind.SERVER,
                context=ctx,
                attributes={
                    'http.method': method,
                    'http.url': str(request.url),
                    'http.route': path,
                    'http.scheme': request.url.scheme,
                    'service.component': service_component,
                },
        ) as span:
            try:
                response = await call_next(request)
                span.set_attribute('http.status_code', response.status_code)
                if response.status_code >= 400:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                return response
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                raise

    return tracing_middleware
