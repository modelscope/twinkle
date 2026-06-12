"""Twinkle Server unified exception hierarchy."""

from __future__ import annotations


class TwinkleServerError(Exception):
    """Base class for all Twinkle Server exceptions."""
    pass


class StateBackendError(TwinkleServerError):
    """State backend operation failed (connection lost, timeout, data serialization error, etc.)."""
    pass


class ConfigError(TwinkleServerError):
    """Invalid configuration value for a known field.

    Used when a field is present and parseable but its value is not in the
    permitted set (e.g. ``backend`` is ``""`` or ``"hf"``). Carries enough
    detail for the operator to find and fix the offending YAML entry without
    re-running the server.
    """

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
    pass


class ResourceExhaustedError(TwinkleServerError):
    """Resource exhausted — queue full, insufficient memory, connection pool exhausted, etc."""
    pass
