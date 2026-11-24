# Copyright (c) ModelScope Contributors. All rights reserved.
"""Record admission decisions independently of their HTTP or future envelope."""

from twinkle.server.exceptions import RateLimitExceededError, ResourceExhaustedError, ResourceQuotaExceededError
from twinkle.server.identity import tenant_fingerprint
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
    record_rejection(reason, deployment=deployment, fingerprint=fingerprint, base_model=base_model)


def record_training_rejection(failure, *, token, base_model):
    reason = {'resource_quota_exceeded': 'quota', 'capacity_unavailable': 'capacity'}.get(failure.reason_code)
    if reason is not None:
        record_rejection(reason, deployment='TrainingController', fingerprint=tenant_fingerprint(token),
                         base_model=base_model)


def record_rejection(reason, *, deployment, fingerprint=None, base_model=None):
    tags = {'deployment': deployment, 'reason': reason}
    if fingerprint:
        tags['tenant_fingerprint'] = fingerprint
    if base_model:
        tags['base_model'] = base_model
    MetricsRegistry.get().admission_rejections.add(1, attributes=tags)
