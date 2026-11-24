"""Twinkle Server tracing utilities — thin wrapper over OpenTelemetry tracing."""

from __future__ import annotations
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from fastapi import Request

try:
    from opentelemetry import context, trace
    from opentelemetry.propagate import extract, inject
    _OTEL_AVAILABLE = True
except Exception:
    _OTEL_AVAILABLE = False

from .correlation import BASE_MODEL, TASK_ID, TENANT_FINGERPRINT, TRAINING_ID, set_correlation_attrs


def inject_context(carrier: dict) -> None:
    """Inject current trace context into carrier. Noop when OTEL is not installed."""
    if not _OTEL_AVAILABLE:
        return
    inject(carrier)


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


def set_span_error(span, error_type: str) -> None:
    if _OTEL_AVAILABLE:
        span.set_attribute('error.type', error_type)
        span.set_status(trace.Status(trace.StatusCode.ERROR))


@contextmanager
def traced_operation(
    name: str,
    *,
    attrs: Mapping[str, Any] | None = None,
    parent_context: Any = None,
    kind=None,
    tracer_name: str = 'twinkle.server.business',
) -> Iterator[Any]:
    """Record the operation and the exception type without exporting exception text."""
    if not _OTEL_AVAILABLE:
        yield _NoopSpan()
        return

    tracer = trace.get_tracer(tracer_name)
    with tracer.start_as_current_span(
        name,
        context=parent_context,
        kind=kind if kind is not None else trace.SpanKind.INTERNAL,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        set_correlation_attrs(span, attrs)
        try:
            yield span
        except Exception as exc:
            error_type = type(exc).__name__
            set_span_error(span, error_type)
            span.add_event('exception', {'exception.type': error_type})
            raise


def trace_http_client(method: str, service_type: str):
    return traced_operation(
        f'{method} {service_type}',
        kind=trace.SpanKind.CLIENT if _OTEL_AVAILABLE else None,
        attrs={'http.request.method': method, 'twinkle.target': service_type},
    )


def record_http_status(span, status: int, *, client: bool = False) -> None:
    span.set_attribute('http.response.status_code', status)
    if status >= (400 if client else 500):
        set_span_error(span, str(status))


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
        method = request.method

        # Continue an upstream trace when the inbound request carries trace
        # headers (e.g. a Gateway -> Model / Gateway -> Sampler hop), so the
        # SERVER span attaches to the propagated context instead of starting a
        # fresh, disconnected trace. A request with no trace headers yields an
        # empty context and a new trace is started normally.
        ctx = extract(request.headers)

        with traced_operation(
                method,
                kind=trace.SpanKind.SERVER,
                parent_context=ctx,
                tracer_name=f'twinkle.server.{service_component}',
                attrs={
                    'http.request.method': method,
                    'url.scheme': request.url.scheme,
                    'service.component': service_component,
                },
        ) as span:
            try:
                response = await call_next(request)
            except Exception:
                span.set_attribute('http.response.status_code', 500)
                raise
            else:
                record_http_status(span, response.status_code)
                return response
            finally:
                route = getattr(request.scope.get('route'), 'path', None)
                if route is not None:
                    span.update_name(f'{method} {route}')
                    span.set_attribute('http.route', route)
                set_correlation_attrs(span, {
                    TENANT_FINGERPRINT: getattr(request.state, 'tenant_fingerprint', None),
                    TRAINING_ID: getattr(request.state, 'training_id', None),
                    TASK_ID: getattr(request.state, 'task_id', None),
                    BASE_MODEL: getattr(request.state, 'base_model', None),
                    'twinkle.http_request_id': getattr(request.state, 'request_id', None),
                })

    return tracing_middleware
