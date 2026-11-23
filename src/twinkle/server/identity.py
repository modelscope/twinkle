# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fingerprint calculation shared by checkpoint paths and telemetry."""
import hmac

CHECKPOINT_SALT_DEFAULT = 'twinkle-path-salt-v1'


def fingerprint(value: str, key: str) -> str:
    return hmac.digest(key.encode('utf-8'), value.encode('utf-8'), 'sha256').hex()
