"""Twinkle Server unified exception hierarchy.

Every exception carries an ``error_code`` (an HTTP-status-shaped int in 400-599) and
a ``category`` (:class:`ErrorCategory`). A single ``TwinkleServerError`` exception
handler (see the gateway/model/sampler apps) reads these two attributes to build a
structured response whose fields sit at the top level of the body -- not nested
under ``detail``.
"""

from __future__ import annotations

from twinkle_client.types.errors import ErrorCategory


class TwinkleServerError(Exception):
    """Base class for all Twinkle Server exceptions.

    ``error_code`` / ``category`` are class-level defaults a subclass overrides; an
    instance may also override them via keyword to avoid a subclass per status code.
    """

    error_code: int = 500
    category: ErrorCategory = ErrorCategory.Server

    def __init__(
        self,
        message: str = '',
        *,
        error_code: int | None = None,
        category: ErrorCategory | None = None,
    ) -> None:
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code
        if category is not None:
            self.category = category


class StateBackendError(TwinkleServerError):
    """State backend operation failed (connection lost, timeout, data serialization error, etc.)."""

    error_code = 500
    category = ErrorCategory.Server


class ConfigError(TwinkleServerError):
    """Invalid configuration value for a known field.

    Used when a field is present and parseable but its value is not in the
    permitted set (e.g. ``backend`` is ``""`` or ``"hf"``). Carries enough
    detail for the operator to find and fix the offending YAML entry without
    re-running the server.
    """

    error_code = 500
    category = ErrorCategory.Server

    def __init__(
        self,
        field: str,
        value: object,
        allowed: list[str] | tuple[str, ...] | None = None,
        message: str | None = None,
    ) -> None:
        self.field = field
        self.value = value
        self.allowed = list(allowed) if allowed is not None else None
        if message is None:
            allowed_part = f', allowed: {self.allowed}' if self.allowed is not None else ''
            message = f'Invalid value for {field}: {value!r}{allowed_part}'
        super().__init__(message)


class ConfigParseError(TwinkleServerError):
    """The configuration source could not be parsed (malformed YAML, ...).

    Distinct from ``pydantic.ValidationError`` (which signals that a parsed
    value violates a field/cross-field rule) and from ``FileNotFoundError``
    (which signals that the source could not be read at all).
    """

    error_code = 500
    category = ErrorCategory.Server


class ResourceExhaustedError(TwinkleServerError):
    """Resource exhausted — queue full, insufficient memory, connection pool exhausted, etc."""

    error_code = 503
    category = ErrorCategory.Server


class EndpointUnavailableError(TwinkleServerError):
    """The endpoint is not implemented by this deployment's backend."""

    error_code = 501
    category = ErrorCategory.Server


class RequestRejectedError(TwinkleServerError):
    """Decision_Boundary-left failure: rejectable from the request body, deployment
    config, and loaded schema alone, so it is returned with a real HTTP status code
    and writes NO future record.

    Named ``RequestRejectedError`` rather than ``RequestValidationError`` to avoid a
    collision with ``fastapi.exceptions.RequestValidationError``. The default is a
    400/User rejection; the subclasses below pin the specific status codes from the
    Decision_Boundary placement table.
    """

    error_code = 400
    category = ErrorCategory.User


class TrainModeMismatchError(RequestRejectedError):
    """The request's train mode does not match the deployment's (LoRA vs full)."""

    error_code = 400
    category = ErrorCategory.User


class InputTokensExceededError(RequestRejectedError):
    """The request's input token count exceeds ``max_input_tokens``."""

    error_code = 422
    category = ErrorCategory.User


class BatchSizeError(RequestRejectedError):
    """Batch size is incompatible with the data world size (too small / not a multiple)."""

    error_code = 422
    category = ErrorCategory.User


class RateLimitExceededError(RequestRejectedError):
    """The request or token rate exceeds the configured limit."""

    error_code = 429
    category = ErrorCategory.User


class ResourceQuotaExceededError(RequestRejectedError):
    """The caller exhausted a configured per-token resource quota."""

    error_code = 429
    category = ErrorCategory.User


class ResourceNotFoundError(RequestRejectedError):
    """A well-formed request names a resource (adapter / session) that is absent.

    Distinct from a malformed request (400): the request itself is valid but the
    referenced resource does not exist or is expiring, so it is a 404 on the
    Decision_Boundary left. Raised (never ``assert``-ed) so the check survives
    ``python -O`` and is classified as user-facing rather than a 500.
    """

    error_code = 404
    category = ErrorCategory.User


class FullModeBusyError(RequestRejectedError):
    """A full-parameter (exclusive) model deployment already has a holder.

    Full-parameter training rewrites the shared base-model weights, so a single
    deployment can only host one training task at a time. It is a request rejection
    (a second tenant is turned away), hence a 409 on the Decision_Boundary left.
    """

    error_code = 409
    category = ErrorCategory.User

    def __init__(self, current_holder: str) -> None:
        self.current_holder = current_holder
        super().__init__('This deployment runs in full-parameter (exclusive) mode and is already '
                         f'held by another training task ({current_holder}). Only one full-parameter '
                         'training task is allowed at a time; retry after it is released.')
