"""Twinkle Server tracing utilities — thin wrapper over OpenTelemetry tracing."""

from __future__ import annotations

from fastapi import Request

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

        with tracer.start_as_current_span(
                span_name,
                kind=trace.SpanKind.SERVER,
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
