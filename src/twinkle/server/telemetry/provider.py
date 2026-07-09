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
from typing import Any

from twinkle.server.config.telemetry import TelemetryConfig
from twinkle.utils import get_logger

logger = get_logger()

# Loggers belonging to the OTLP transport stack. Their own records must never
# be routed back through the OTLP LoggingHandler: an exporter error logged
# under ``opentelemetry.*`` (or its gRPC / urllib3 transport) would be
# re-handled and re-exported, amplifying into a feedback loop. The filter below
# drops records originating from these logger trees.
_OTLP_TRANSPORT_LOGGER_PREFIXES = ('opentelemetry', 'grpc', 'urllib3')


class _OTLPTransportFilter(logging.Filter):
    """Drop log records emitted by the OTLP transport stack (feedback-loop guard)."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - logging API
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
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
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

# Logging instrumentor is also optional.
try:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor

    _LOGGING_INSTRUMENTOR_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    _LOGGING_INSTRUMENTOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level state for shutdown.
# ---------------------------------------------------------------------------
_tracer_provider: Any | None = None
_meter_provider: Any | None = None
_logger_provider: Any | None = None
_logging_handler: Any | None = None
_initialized: bool = False


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


def init_telemetry(config: TelemetryConfig) -> None:
    """Initialize the three OTEL pillars (traces, metrics, logs).

    No-op when ``config.enabled`` is False or the OTEL SDK is missing.
    """
    global _tracer_provider, _meter_provider, _logger_provider
    global _logging_handler, _initialized

    if not config.enabled:
        return

    if not _OTEL_AVAILABLE:
        logger.warning(
            'OpenTelemetry SDK not available, skipping telemetry init: %s',
            _OTEL_IMPORT_ERROR,
        )
        return

    if _initialized:
        logger.debug('Telemetry already initialized; skipping re-init.')
        return

    # ---- Resource -------------------------------------------------------
    resource_attrs: dict[str, str] = {'service.name': config.service_name}
    if config.resource_attributes:
        resource_attrs.update(config.resource_attributes)
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
    trace.set_tracer_provider(tracer_provider)
    _tracer_provider = tracer_provider

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
    )
    metrics.set_meter_provider(meter_provider)
    _meter_provider = meter_provider

    # ---- Logs -----------------------------------------------------------
    if use_console:
        log_exporter = ConsoleLogExporter()
    else:
        log_exporter = OTLPLogExporter(endpoint=config.otlp_endpoint)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)
    _logger_provider = logger_provider

    # Bridge Python logging -> OTEL logs.
    if _LOGGING_INSTRUMENTOR_AVAILABLE:
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('LoggingInstrumentor failed to instrument: %s', exc)

    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    # Drop OTLP transport-stack records BEFORE they reach the OTLP exporter, so
    # an exporter error logged under ``opentelemetry.*`` / ``grpc`` / ``urllib3``
    # is not re-handled and re-exported into a feedback loop. Attach the filter
    # to the handler before the handler is added to any logger.
    handler.addFilter(_OTLPTransportFilter())
    # Attach to BOTH the root logger and the ``twinkle`` namespace logger.
    # ``twinkle.utils.logger`` configures the ``twinkle`` logger with
    # ``propagate=False`` and its own StreamHandler, so log records emitted
    # under ``twinkle.*`` (which is the entire server codebase) never bubble
    # up to root and would be invisible to an OTLP handler bound there only.
    logging.getLogger().addHandler(handler)
    logging.getLogger('twinkle').addHandler(handler)
    _logging_handler = handler

    _initialized = True
    logger.info(
        'Telemetry initialized (service=%s, debug=%s, otlp_endpoint=%s)',
        config.service_name,
        config.debug,
        config.otlp_endpoint,
    )


def shutdown_telemetry() -> None:
    """Shutdown all OTEL providers and detach the logging handler."""
    global _tracer_provider, _meter_provider, _logger_provider
    global _logging_handler, _initialized

    if _logging_handler is not None:
        for logger_name in ('', 'twinkle'):
            try:
                logging.getLogger(logger_name).removeHandler(_logging_handler)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning('Failed to detach logging handler from %r: %s', logger_name, exc)
        _logging_handler = None

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('TracerProvider shutdown failed: %s', exc)
        _tracer_provider = None

    if _meter_provider is not None:
        try:
            _meter_provider.shutdown()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('MeterProvider shutdown failed: %s', exc)
        _meter_provider = None

    if _logger_provider is not None:
        try:
            _logger_provider.shutdown()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('LoggerProvider shutdown failed: %s', exc)
        _logger_provider = None

    _initialized = False


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
