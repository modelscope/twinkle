"""Configuration signature validation for state persistence integrity."""
from __future__ import annotations

import hashlib
import json
import logging
from enum import Enum
from typing import Any

from twinkle.server.exceptions import ConfigMismatchError
from twinkle.server.state.backend.base import StateBackend

logger = logging.getLogger(__name__)

_SIGNATURE_KEY = '_meta::config_signature'


class SignatureMismatchPolicy(str, Enum):
    """Policy for handling config signature mismatches."""
    WARN = 'warn'  # Log warning and continue
    CLEAR = 'clear'  # Clear all backend data and continue
    ABORT = 'abort'  # Raise error, refuse to start


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
        logger.info('No previous config signature found. Storing current signature.')
        await backend.set(_SIGNATURE_KEY, current_sig)
        return True

    if stored_sig == current_sig:
        logger.debug('Config signature matches stored value.')
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
        logger.warning('Clearing all backend data due to config signature mismatch.')
        all_keys = await backend.keys('*')
        for key in all_keys:
            if not key.startswith('_meta::'):
                await backend.delete(key)
        await backend.set(_SIGNATURE_KEY, current_sig)
        return False

    elif policy == SignatureMismatchPolicy.ABORT:
        raise ConfigMismatchError(f"Configuration signature mismatch. "
                                  f"Stored: {stored_sig[:12]}..., Current: {current_sig[:12]}... "
                                  f"Use policy='warn' or 'clear' to allow startup with changed config.")

    return False


# ---------------------------------------------------------------------------
# CLI startup hook (R15)
# ---------------------------------------------------------------------------


def _format_diff(stored: dict[str, Any] | None, current: dict[str, Any]) -> str:
    """Render a stored-vs-current diff suitable for a remediation hint."""
    lines: list[str] = []
    keys = sorted(set((stored or {}).keys()) | set(current.keys()))
    for k in keys:
        s = (stored or {}).get(k, '<absent>')
        c = current.get(k, '<absent>')
        if s != c:
            lines.append(f'  - {k}: stored={s!r} current={c!r}')
    return '\n'.join(lines) if lines else '  (no field-level diff — values differ at nested level)'


async def validate_against_backend(persistence_config: Any, current_config: dict[str, Any]) -> None:
    """Validate the persistence config signature on launcher startup (R15).

    Builds a backend from ``persistence_config`` (a :class:`PersistenceConfig`),
    computes the current signature, and compares it to the stored value:
    - if no signature is stored, store the current one and continue (R15.4);
    - if signatures match, return cleanly;
    - if they differ, raise :class:`ConfigMismatchError` with a stored-vs-
      current diff and a remediation hint (R15.2, R15.3).

    Designed to be called BEFORE ``ray.init`` so the launcher can fail fast
    without spinning up the cluster (R15.1).
    """
    from twinkle.server.state.backend.factory import create_backend

    backend = create_backend(persistence_config)
    current_sig = compute_signature(current_config)
    stored_sig = await backend.get(_SIGNATURE_KEY)

    if stored_sig is None:
        await backend.set(_SIGNATURE_KEY, current_sig)
        logger.info('No previous config signature found. Stored current signature.')
        return

    if stored_sig == current_sig:
        return

    stored_payload = await backend.get('_meta::config_payload')
    diff = _format_diff(stored_payload if isinstance(stored_payload, dict) else None, current_config)
    raise ConfigMismatchError('Persistence configuration drifted since the last launch. '
                              f'Stored signature: {stored_sig[:12]}..., current signature: {current_sig[:12]}...\n'
                              f'Differences:\n{diff}\n'
                              'Remediation: either revert the persistence section to match the stored '
                              'value, or clear the persisted state with '
                              '`python -m twinkle.server clear persistence --config <yaml>` and relaunch.')
