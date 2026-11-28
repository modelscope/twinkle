"""OpenTelemetry provider initialization for Twinkle server.

Bootstraps the three OTEL pillars (traces, metrics, logs) with either an OTLP
or a console exporter. Designed to be a thin, side-effect-driven module that
exposes:

- ``TelemetryConfig``: pydantic configuration model
- ``init_telemetry``: entry point that wires up global providers
- ``shutdown_telemetry``: graceful teardown for the global providers
- ``get_meter``: convenience accessor used by ``MetricsRegistry``
"""

from __future__ import annotations
import logging
import uuid
from copy import copy
from typing import Any

from twinkle.utils import get_logger
from twinkle_server.config.telemetry import TelemetryConfig
from .correlation import telemetry_node

logger = get_logger()

# Loggers belonging to the OTLP transport stack. Their own records must never
# be routed back through the OTLP LoggingHandler: an exporter error logged
# under ``opentelemetry.*`` (or its gRPC / urllib3 transport) would be
# re-handled and re-exported, amplifying into a feedback loop. The filter below
# drops records originating from these logger trees.
_OTLP_TRANSPORT_LOGGER_PREFIXES = ('opentelemetry', 'grpc', 'urllib3')


class _OTLPTransportFilter(logging.Filter):
    """Drop log records emitted by the OTLP transport stack (feedback-loop guard)."""

    def filter(self, record: logging.LogRecord) -> bool:  # logging API
        name = record.name or ''
        for prefix in _OTLP_TRANSPORT_LOGGER_PREFIXES:
            if name == prefix or name.startswith(prefix + '.'):
                return False
        return True


# ---------------------------------------------------------------------------
# Optional OTEL imports — keep them lazy/guarded so that a missing optional
# dependency does not break the rest of the server.
# ---------------------------------------------------------------------------
try:
    from opentelemetry import metrics, trace
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.metrics import Observation
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _OTEL_AVAILABLE = True
    _OTEL_IMPORT_ERROR: BaseException | None = None
except Exception as exc:  # pragma: no cover - defensive fallback
    _OTEL_AVAILABLE = False
    _OTEL_IMPORT_ERROR = exc

# OTLP exporters are a separate optional dependency from the SDK itself.
try:
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _OTLP_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    _OTLP_AVAILABLE = False

if _OTEL_AVAILABLE:
    class _TelemetryLoggingHandler(LoggingHandler):
        def emit(self, record: logging.LogRecord) -> None:
            if exc_type := record.exc_info and record.exc_info[0]:
                # Preserve local traceback formatting. Export only structured exception types.
                record = copy(record)
                record.msg = 'Operation failed'
                record.args = ()
                record.error_type = exc_type.__name__
                record.exc_info = None
                record.exc_text = None
            super().emit(record)


# All buckets and observations use seconds.
_LATENCY_BUCKETS = {
    'twinkle.http.request.duration_seconds': (
        .001, .005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10, 30, 60, 120),
    'twinkle.task.wait_seconds': (
        .001, .005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10, 30, 60, 120, 300),
    'twinkle.task.execution_seconds': (
        .005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800),
}


# ---------------------------------------------------------------------------
# Module-level state for shutdown.
# ---------------------------------------------------------------------------
_tracer_provider: Any | None = None
_meter_provider: Any | None = None
_logger_provider: Any | None = None
_logging_handler: Any | None = None
_shutdown = False


class _LoggingWriter:
    """IO adapter that routes writes to Python logging.

    Used to redirect ConsoleExporter output through the logging system
    so that Ray Serve worker logs are properly captured. Without this,
    ConsoleSpanExporter / ConsoleMetricExporter write directly to
    ``sys.stdout`` which Ray reroutes into its internal log files,
    making telemetry output invisible to the standard logging pipeline.
    """

    def __init__(self, logger_name: str = 'twinkle.server.telemetry.export'):
        self._logger = logging.getLogger(logger_name)

    def write(self, text: str) -> int:
        text = text.strip()
        if text:
            self._logger.info(text)
        return len(text)

    def flush(self) -> None:
        pass


def init_telemetry(config: TelemetryConfig) -> bool:
    """Initialize the three OTEL pillars (traces, metrics, logs).

    Return whether the process has a telemetry pipeline.
    """
    global _tracer_provider, _meter_provider, _logger_provider
    global _logging_handler

    if not config.enabled:
        return False

    if not _OTEL_AVAILABLE:
        logger.warning(
            'OpenTelemetry SDK not available, skipping telemetry init: %s',
            _OTEL_IMPORT_ERROR,
        )
        return False

    if _shutdown:
        logger.warning("Telemetry has shut down in this process; start a new process to enable it again")
        return False

    if _logging_handler is not None:
        return True

    # ---- Resource -------------------------------------------------------
    resource_attrs: dict[str, str] = {'service.name': config.service_name}
    if config.resource_attributes:
        resource_attrs.update(config.resource_attributes)
    resource_attrs['twinkle.node'] = telemetry_node()
    # OTLP backends need separate cumulative time series before Grafana sums replicas.
    resource_attrs['service.instance.id'] = uuid.uuid4().hex
    resource = Resource.create(resource_attrs)

    use_console = config.debug or not _OTLP_AVAILABLE
    if config.debug is False and not _OTLP_AVAILABLE:
        logger.warning('OTLP exporters not available; falling back to console exporters.')

    # When using console exporters, route their output through the Python
    # logging system so that Ray Serve workers actually surface the data.
    _console_writer = _LoggingWriter() if use_console else None

    # ---- Traces ---------------------------------------------------------
    if use_console:
        span_exporter = ConsoleSpanExporter(out=_console_writer)
    else:
        span_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    # ---- Metrics --------------------------------------------------------
    if use_console:
        metric_exporter = ConsoleMetricExporter(out=_console_writer)
    else:
        metric_exporter = OTLPMetricExporter(endpoint=config.otlp_endpoint)

    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=config.export_interval_ms,
    )
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader],
        views=[View(instrument_name=name, aggregation=ExplicitBucketHistogramAggregation(bounds))
               for name, bounds in _LATENCY_BUCKETS.items()],
    )
    meter_provider.get_meter('twinkle-server').create_observable_gauge(
        'twinkle.telemetry.heartbeat',
        callbacks=[lambda _: [Observation(1, {'node': resource_attrs['twinkle.node']})]],
        description='Runtime metrics exporter heartbeat',
    )

    # ---- Logs -----------------------------------------------------------
    if use_console:
        log_exporter = ConsoleLogExporter()
    else:
        log_exporter = OTLPLogExporter(endpoint=config.otlp_endpoint)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    set_logger_provider(logger_provider)
    _tracer_provider = tracer_provider
    _meter_provider = meter_provider
    _logger_provider = logger_provider

    handler = _TelemetryLoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    # Drop OTLP transport-stack records BEFORE they reach the OTLP exporter, so
    # an exporter error logged under ``opentelemetry.*`` / ``grpc`` / ``urllib3``
    # is not re-handled and re-exported into a feedback loop. Attach the filter
    # to the handler before the handler is added to any logger.
    handler.addFilter(_OTLPTransportFilter())
    # Server loggers retain their twinkle.server.* names after the package move,
    # preserving LOG_LEVEL inheritance, console handlers and telemetry scopes.
    # Attach to BOTH the root logger and the ``twinkle`` namespace logger.
    # ``twinkle.utils.logger`` configures the ``twinkle`` logger with
    # ``propagate=False`` and its own StreamHandler, so log records emitted
    # under ``twinkle.*`` never bubble
    # up to root and would be invisible to an OTLP handler bound there only.
    logging.getLogger().addHandler(handler)
    twinkle_logger = logging.getLogger('twinkle')
    if not twinkle_logger.propagate:
        twinkle_logger.addHandler(handler)
    _logging_handler = handler

    logger.info(
        'Telemetry initialized (service=%s, debug=%s, otlp_endpoint=%s)',
        config.service_name,
        config.debug,
        config.otlp_endpoint,
    )
    return True


def shutdown_telemetry() -> None:
    """Close this process's pipeline. OTel global providers can only be set once."""
    global _tracer_provider, _meter_provider, _logger_provider
    global _logging_handler, _shutdown

    if _logging_handler is None:
        return

    _shutdown = True
    for logger_name in ('', 'twinkle'):
        logging.getLogger(logger_name).removeHandler(_logging_handler)
    _logging_handler = None
    for provider in (_tracer_provider, _meter_provider, _logger_provider):
        try:
            provider.shutdown()
        except Exception as exc:  # pragma: no cover - exporter failures must not interrupt shutdown
            logger.warning('Telemetry provider shutdown failed (%s)', type(exc).__name__)
    _tracer_provider = _meter_provider = _logger_provider = None


class _NoopInstrument:
    """No-op instrument for when OTEL SDK is not available."""

    def add(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass


class _NoopMeter:
    """No-op meter for when OTEL SDK is not available."""

    def create_counter(self, *args, **kwargs):
        return _NoopInstrument()

    def create_up_down_counter(self, *args, **kwargs):
        return _NoopInstrument()

    def create_histogram(self, *args, **kwargs):
        return _NoopInstrument()

    def create_observable_gauge(self, *args, **kwargs):
        return _NoopInstrument()


_noop_meter = _NoopMeter()


def get_meter(name: str = 'twinkle-server'):
    """Return an OTEL meter. Returns NoOp meter if OTEL SDK is not available."""
    if not _OTEL_AVAILABLE:
        return _noop_meter
    return metrics.get_meter(name)
