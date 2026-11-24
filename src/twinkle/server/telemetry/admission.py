# Copyright (c) ModelScope Contributors. All rights reserved.
"""Record admission decisions independently of their HTTP or future envelope."""

from twinkle.server.exceptions import RateLimitExceededError, ResourceExhaustedError, ResourceQuotaExceededError
from twinkle.server.resources.allocation import CapacityUnavailableError
from twinkle.server.telemetry.metrics import MetricsRegistry


def record_admission_rejection(error, *, deployment, fingerprint=None, base_model=None):
    if isinstance(error, RateLimitExceededError):
        reason = 'rate_limit'
    elif isinstance(error, ResourceQuotaExceededError):
        reason = 'quota'
    elif isinstance(error, (CapacityUnavailableError, ResourceExhaustedError)):
        reason = 'capacity'
    else:
        return
    tags = {'deployment': deployment, 'reason': reason}
    if fingerprint:
        tags['tenant_fingerprint'] = fingerprint
    if base_model:
        tags['base_model'] = base_model
    MetricsRegistry.get().admission_rejections.add(1, attributes=tags)
