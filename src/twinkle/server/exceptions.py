"""Twinkle Server unified exception hierarchy."""

from __future__ import annotations


class TwinkleServerError(Exception):
    """Base class for all Twinkle Server exceptions."""
    pass


class StateBackendError(TwinkleServerError):
    """State backend operation failed (connection lost, timeout, data serialization error, etc.)."""
    pass


class ConfigMismatchError(TwinkleServerError):
    """Configuration signature mismatch — config changes detected after restart, persisted data may be incompatible with current configuration."""
    pass


class ResourceExhaustedError(TwinkleServerError):
    """Resource exhausted — queue full, insufficient memory, connection pool exhausted, etc."""
    pass
