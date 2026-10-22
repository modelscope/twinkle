# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-deployment metric adapters must not survive ``MetricsRegistry.reset()``.

``ensure_telemetry_initialized`` resets the registry *in order to* rebind instruments to a
real MeterProvider; an adapter cached at module level survived that and kept recording into
NoOp instruments for the life of the process. This test fails before (module-level
cache) and passes once the caches live on the registry instance.
"""
from twinkle.server.telemetry.metrics import MetricsRegistry, get_task_metrics


def test_task_metrics_rebind_after_registry_reset():
    first = get_task_metrics('Model')
    MetricsRegistry.reset()
    second = get_task_metrics('Model')
    assert first is not second
    # The point is the *instrument*, not the wrapper identity: assert the bound
    # instrument object differs, so a cheap "return a new wrapper around the same
    # instrument" implementation cannot pass.
    assert first.execution_seconds._instrument is not second.execution_seconds._instrument
