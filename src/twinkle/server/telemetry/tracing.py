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
    """获取 tracer 实例。OTEL 未安装时返回 NoOp tracer。"""
    if not _OTEL_AVAILABLE:
        return _NoopTracer()
    return trace.get_tracer(name)


def inject_context(carrier: dict) -> None:
    """将当前 trace context 注入到 carrier。OTEL 未安装时为 noop。"""
    if not _OTEL_AVAILABLE:
        return
    inject(carrier)


def extract_context(carrier: dict):
    """从 carrier 中提取 trace context。OTEL 未安装时返回空 context。"""
    if not _OTEL_AVAILABLE:
        return None
    return extract(carrier)


def get_current_span():
    """获取当前活跃的 span。OTEL 未安装时返回 noop span。"""
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
