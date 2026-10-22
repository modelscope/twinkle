# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reject a request this deployment cannot serve, before it is enqueued.

Two questions, both answerable from the request body plus the deployment's own
configuration:

1. does the body set a parameter that only exists on the *other* backend?
2. does the endpoint exist on this backend at all?

Both used to be answered by the backend raising during execution -- which means after
the task was enqueued, a future record written, and the call fanned out to every
data-parallel rank. Answering them here makes the number of ranks that ran the backend
method for a rejected request exactly zero.

**Why there is no spelling check over passthrough keys.** One was implemented here and
removed after it rejected a working call: ``set_processor('InputProcessor',
padding_side='right')``, where ``padding_side`` is a real parameter that
``InputProcessor`` reads via ``kwargs.get('padding_side')`` and therefore never declares.
It scored 0.75 against the declared ``padding_free``. That is not a tunable threshold
problem -- ``inspect.signature`` cannot see a ``**kwargs`` read at all, so "misspelled"
and "read out of ``**kwargs``" are indistinguishable to it, and any threshold catching
``bate`` -> ``beta`` also catches this. A check that rejects valid requests is worse than
no check, so the passthrough region is forwarded as given and a misspelled plugin
argument still surfaces from the plugin itself.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Optional

from twinkle.protocol.types.base import FieldRole, fields_with_role, read_backend_only
from twinkle.server.exceptions import EndpointUnavailableError, RequestRejectedError
from twinkle.utils.logger import get_logger


class BackendCapability(StrEnum):
    """An endpoint that not every backend implements.

    Only the gradient-path splits belong here. Megatron fuses forward and backward, so
    it raises ``NotImplementedError('Megatron only supports forward_backward and
    forward_only')`` for each of these three; naming them makes that a 501 with the
    alternative endpoints in the message instead of a 500 carrying a backend traceback.
    """

    Forward = 'forward'
    Backward = 'backward'
    CalculateLoss = 'calculate_loss'


# Capabilities a backend does NOT provide. Absent backends support everything; ``mock``
# is deliberately absent because it is a test double that accepts every call.
_UNSUPPORTED: dict[str, frozenset[str]] = {
    'megatron': frozenset({BackendCapability.Forward, BackendCapability.Backward, BackendCapability.CalculateLoss}),
}

_ALTERNATIVES = 'use `forward_backward` (training) or `forward_only` (inference) instead'


def resolve_backend(service: Any) -> str | None:
    """This deployment's declared backend, or ``None`` when it has no backend concept.

    Read from the deployment's own configuration (``ModelManagement.backend``), never
    inferred from the wrapper class name, the request body, or a backend exception: those
    are all restatements of the same fact one step further from the source, and the
    sampler deployment has no ``backend`` at all.
    """
    backend = getattr(service, 'backend', None)
    if backend is None:
        # Not an error: Sampler deployments have no ``backend`` attribute at all. But a
        # silent ``None`` meant the whole backend-compat preflight vanished with no
        # trace, so the skip is now observable. ``debug`` not ``warning``:
        # for Sampler the skip is normal and happens every request.
        get_logger().debug('backend-compat preflight skipped: %s exposes no ``backend`` attribute',
                           type(service).__name__)
        return None
    return backend if isinstance(backend, str) else None


def assert_endpoint_available(service: Any, capability: str | None) -> None:
    """Raise 501 when this deployment's backend does not implement ``capability``."""
    if capability is None:
        return
    backend = resolve_backend(service)
    if backend is not None and capability in _UNSUPPORTED.get(backend, frozenset()):
        raise EndpointUnavailableError(f'`{capability}` is not available on the {backend} backend; {_ALTERNATIVES}.')


def assert_backend_fields(service: Any, body: Any) -> None:
    """Raise 422 for a field restricted to a backend other than this deployment's.

    Only a field carrying a non-``None`` value is checked. A restricted field is always
    ``Optional[...] = None``, precisely so that "not sent" and "sent to the wrong
    backend" stay distinguishable -- were it given the backend's own default, every
    request on the other half of the fleet would be rejected.
    """
    backend = resolve_backend(service)
    if backend is None or backend == 'mock':
        return
    offenders = []
    for name, info in fields_with_role(type(body), FieldRole.BackendKwarg).items():
        allowed = read_backend_only(info)
        if allowed and backend not in allowed and getattr(body, name, None) is not None:
            offenders.append((name, allowed))
    if offenders:
        details = '; '.join(f'`{name}` is only supported on {"/".join(allowed)}' for name, allowed in offenders)
        raise RequestRejectedError(
            f'This deployment runs the {backend} backend. {details}. Remove the parameter or target a '
            f'deployment running a supporting backend.',
            error_code=422)


def assert_request_supported(service: Any, body: Any, *, capability: str | None = None) -> None:
    """The single preflight entry point, called from ``run_submit`` before the enqueue.

    Both checks read *declared* metadata, so neither can produce a false positive. A
    spelling heuristic over passthrough keys was tried here and removed: signature
    reflection cannot distinguish a misspelling from a parameter a target reads straight
    out of ``**kwargs``, so it rejected working calls. See the module docstring.
    """
    assert_endpoint_available(service, capability)
    assert_backend_fields(service, body)
