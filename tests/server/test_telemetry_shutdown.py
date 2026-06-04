# Copyright (c) ModelScope Contributors. All rights reserved.
"""Telemetry-flush-on-shutdown regression tests (Requirement 21).

Pins that buffered telemetry is flushed when the server shuts down:
- the launcher invokes the telemetry flush AFTER ``serve.shutdown()``;
- the shared ``flush_telemetry_safely`` helper (used by both the launcher and
  every worker lifespan) calls ``shutdown_telemetry`` and swallows any error so
  a telemetry-shutdown failure never masks the user-facing shutdown path.
"""
from __future__ import annotations

from unittest import mock

from twinkle.server.config import ServerConfig
from twinkle.server.launcher import ServerLauncher


def test_flush_telemetry_safely_calls_shutdown() -> None:
    from twinkle.server.telemetry import worker_init

    with mock.patch('twinkle.server.telemetry.shutdown_telemetry') as shutdown_spy:
        worker_init.flush_telemetry_safely()
    assert shutdown_spy.call_count == 1


def test_flush_telemetry_safely_swallows_errors() -> None:
    from twinkle.server.telemetry import worker_init

    with mock.patch('twinkle.server.telemetry.shutdown_telemetry', side_effect=RuntimeError('boom')):
        # Must not raise.
        worker_init.flush_telemetry_safely()


def test_launcher_shutdown_flushes_telemetry_after_serve_shutdown() -> None:
    """``ServerLauncher._shutdown`` calls the telemetry flush after Serve is
    torn down, in that order."""
    launcher = ServerLauncher(config=ServerConfig())
    order: list[str] = []

    fake_serve = mock.MagicMock()
    fake_serve.shutdown.side_effect = lambda: order.append('serve')

    with mock.patch.dict('sys.modules', {'ray': mock.MagicMock(serve=fake_serve)}):
        with mock.patch('ray.serve', fake_serve, create=True):
            with mock.patch(
                    'twinkle.server.telemetry.flush_telemetry_safely',
                    side_effect=lambda: order.append('telemetry'),
            ) as flush_spy:
                launcher._shutdown()

    assert flush_spy.call_count == 1
    assert order == ['serve', 'telemetry'], f'unexpected shutdown order: {order}'


def test_launcher_shutdown_flushes_even_if_serve_shutdown_raises() -> None:
    """A Serve shutdown error must not prevent the telemetry flush."""
    launcher = ServerLauncher(config=ServerConfig())

    fake_serve = mock.MagicMock()
    fake_serve.shutdown.side_effect = RuntimeError('serve boom')

    with mock.patch('ray.serve', fake_serve, create=True):
        with mock.patch('twinkle.server.telemetry.flush_telemetry_safely') as flush_spy:
            launcher._shutdown()

    assert flush_spy.call_count == 1
