# Copyright (c) ModelScope Contributors. All rights reserved.
"""User fingerprints for the current authenticated request."""

import os
from contextvars import ContextVar

from twinkle.server.identity import CHECKPOINT_SALT_DEFAULT, fingerprint

USER_ID_KEY_ENV = "TWINKLE_TELEMETRY_USER_ID_KEY"
current_user_id: ContextVar[str | None] = ContextVar("twinkle_user_id", default=None)


def user_id_key() -> str | None:
    key = os.getenv(USER_ID_KEY_ENV)
    if key and key != os.getenv("TWINKLE_TOKEN_SALT", CHECKPOINT_SALT_DEFAULT):
        return key
    return None


def user_fingerprint(token: str | None, client_id: str | None = None) -> str | None:
    key = user_id_key()
    if key:
        value = f"token:{token}" if token else f"client:{client_id}" if client_id else "anonymous"
        return fingerprint(value, key)
    return None
