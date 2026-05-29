"""Configuration signature validation for state persistence integrity."""
from __future__ import annotations

import hashlib
import json
import logging
from enum import Enum
from typing import Any

from twinkle.server.state.backend.base import StateBackend

logger = logging.getLogger(__name__)

_SIGNATURE_KEY = "_meta::config_signature"


class SignatureMismatchPolicy(str, Enum):
    """Policy for handling config signature mismatches."""
    WARN = "warn"  # Log warning and continue
    CLEAR = "clear"  # Clear all backend data and continue
    ABORT = "abort"  # Raise error, refuse to start


def compute_signature(config: dict[str, Any]) -> str:
    """Compute a SHA256 hash of the configuration dictionary.

    Args:
        config: Configuration dictionary to hash.

    Returns:
        Hex string of SHA256 hash.
    """
    # Sort keys for deterministic serialization
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


async def validate_config_signature(
    backend: StateBackend,
    current_config: dict[str, Any],
    policy: SignatureMismatchPolicy = SignatureMismatchPolicy.WARN,
) -> bool:
    """Validate configuration signature against stored value.

    Compares the current config's hash with the previously stored hash.
    On first run (no stored hash), stores the current hash and returns True.

    Args:
        backend: State backend to read/write signature.
        current_config: Current configuration dictionary.
        policy: Action to take on mismatch.

    Returns:
        True if signature matches or is new. False if mismatch with WARN/CLEAR policy.

    Raises:
        ConfigMismatchError: If policy is ABORT and signature doesn't match.
    """
    current_sig = compute_signature(current_config)
    stored_sig = await backend.get(_SIGNATURE_KEY)

    if stored_sig is None:
        # First run — store signature
        logger.info("No previous config signature found. Storing current signature.")
        await backend.set(_SIGNATURE_KEY, current_sig)
        return True

    if stored_sig == current_sig:
        logger.debug("Config signature matches stored value.")
        return True

    # Mismatch detected
    logger.warning(f"Config signature mismatch! "
                   f"Stored: {stored_sig[:12]}..., Current: {current_sig[:12]}... "
                   f"Policy: {policy.value}")

    if policy == SignatureMismatchPolicy.WARN:
        # Update to new signature and continue
        await backend.set(_SIGNATURE_KEY, current_sig)
        return False

    elif policy == SignatureMismatchPolicy.CLEAR:
        # Clear all data except meta keys, store new signature
        logger.warning("Clearing all backend data due to config signature mismatch.")
        all_keys = await backend.keys("*")
        for key in all_keys:
            if not key.startswith("_meta::"):
                await backend.delete(key)
        await backend.set(_SIGNATURE_KEY, current_sig)
        return False

    elif policy == SignatureMismatchPolicy.ABORT:
        from twinkle.server.exceptions import ConfigMismatchError
        raise ConfigMismatchError(
            f"Configuration signature mismatch. "
            f"Stored: {stored_sig[:12]}..., Current: {current_sig[:12]}... "
            f"Use policy='warn' or 'clear' to allow startup with changed config.")

    return False
