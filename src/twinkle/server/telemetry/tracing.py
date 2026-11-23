"""Twinkle Server tracing utilities — thin wrapper over OpenTelemetry tracing."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from fastapi import Request
from typing import Any

try:
    from opentelemetry import context, trace
    from opentelemetry.propagate import extract, inject
    _OTEL_AVAILABLE = True
except Exception:
    _OTEL_AVAILABLE = False

from .correlation import MODEL_ID, USER_ID, set_correlation_attrs


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


def get_trace_context():
    return context.get_current() if _OTEL_AVAILABLE else None


def get_trace_id(parent_context: Any = None) -> str | None:
    if _OTEL_AVAILABLE:
        span_context = trace.get_current_span(parent_context).get_span_context()
        if span_context.is_valid and span_context.trace_flags.sampled:
            return f'{span_context.trace_id:032x}'
    return None


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
    parent_context: Any = None,
    tracer_name: str = 'twinkle.server.business',
) -> Iterator[Any]:
    """Run a business-layer block under one OTEL span.

    OpenTelemetry records exceptions and closes the span on exit. Without
    the SDK, the context manager yields a NoOp span.
    """
    if not _OTEL_AVAILABLE:
        yield _NoopSpan()
        return

    tracer = trace.get_tracer(tracer_name)
    with tracer.start_as_current_span(name, context=parent_context) as span:
        if attrs:
            set_correlation_attrs(span, attrs)
        yield span


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

        # Continue an upstream trace when the inbound request carries trace
        # headers (e.g. a Gateway -> Model / Gateway -> Sampler hop), so the
        # SERVER span attaches to the propagated context instead of starting a
        # fresh, disconnected trace. A request with no trace headers yields an
        # empty context and a new trace is started normally.
        ctx = extract(request.headers)

        with tracer.start_as_current_span(
                method,
                kind=trace.SpanKind.SERVER,
                context=ctx,
                attributes={
                    'http.method': method,
                    'http.scheme': request.url.scheme,
                    'service.component': service_component,
                },
        ) as span:
            try:
                response = await call_next(request)
                route = getattr(request.scope.get('route'), 'path', None)
                if route is not None:
                    span.update_name(f'{method} {route}')
                    span.set_attribute('http.route', route)
                set_correlation_attrs(span, {
                    USER_ID: getattr(request.state, 'user_id', None),
                    MODEL_ID: getattr(request.state, 'model_id', None),
                    'twinkle.http_request_id': getattr(request.state, 'request_id', None),
                })
                span.set_attribute('http.status_code', response.status_code)
                if response.status_code >= 400:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                return response
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                raise

    return tracing_middleware
