# Copyright (c) ModelScope Contributors. All rights reserved.
"""Record admission decisions independently of their HTTP or future envelope."""

from twinkle.server.telemetry.metrics import MetricsRegistry


def record_admission_rejection(
    reason: str, *, deployment: str, fingerprint: str | None = None, base_model: str | None = None
) -> None:
    tags = {'deployment': deployment, 'reason': reason}
    if fingerprint:
        tags['tenant_fingerprint'] = fingerprint
    if base_model:
        tags['base_model'] = base_model
    MetricsRegistry.get().admission_rejections.add(1, attributes=tags)
