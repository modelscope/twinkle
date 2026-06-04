# Copyright (c) ModelScope Contributors. All rights reserved.
"""OTLP logging feedback-loop suppression regression tests (Requirement 22).

Pins that the OTLP ``LoggingHandler`` carries a filter dropping records from
the OTLP transport stack (``opentelemetry`` / ``grpc`` / ``urllib3``), so an
exporter error logged under those trees is not re-handled and re-exported into
an amplifying feedback loop, while ordinary application records still flow.
"""
from __future__ import annotations

import logging

from twinkle.server.telemetry.provider import _OTLPTransportFilter


def _record(name: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=logging.ERROR, pathname=__file__, lineno=1, msg='m', args=(), exc_info=None)


def test_filter_drops_transport_stack_records() -> None:
    f = _OTLPTransportFilter()
    for name in (
            'opentelemetry',
            'opentelemetry.exporter.otlp.proto.grpc',
            'grpc',
            'grpc._channel',
            'urllib3',
            'urllib3.connectionpool',
    ):
        assert f.filter(_record(name)) is False, f'expected {name!r} to be dropped'


def test_filter_keeps_application_records() -> None:
    f = _OTLPTransportFilter()
    for name in ('twinkle', 'twinkle.server.state', 'myapp', 'opentelemetryx', 'grpcfoo'):
        assert f.filter(_record(name)) is True, f'expected {name!r} to be kept'
