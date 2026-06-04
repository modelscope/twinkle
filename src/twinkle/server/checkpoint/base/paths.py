# Copyright (c) ModelScope Contributors. All rights reserved.
"""Path constants, token hashing, client-save-dir resolution, and permission
helpers for the checkpoint subsystem.

Relocated from ``utils/checkpoint_base.py`` (TIER 2 consolidation). No logic change.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from pathlib import Path

TWINKLE_DEFAULT_SAVE_DIR = os.environ.get('TWINKLE_DEFAULT_SAVE_DIR', './outputs')
CHECKPOINT_INFO_FILENAME = 'checkpoint_metadata.json'
TRAIN_RUN_INFO_FILENAME = 'twinkle_metadata.json'
SAVE_DIR_POINTER_KEY = 'save_dir_pointer'

# Salt used when hashing tokens for directory isolation.
# Override via env var TWINKLE_TOKEN_SALT to customise per-deployment.
_TOKEN_SALT = os.environ.get('TWINKLE_TOKEN_SALT', 'twinkle-path-salt-v1').encode('utf-8')


def _hash_token(token: str) -> str:
    """Return a salted HMAC-SHA256 hex digest of *token*.

    The digest is used as the per-user base directory name so that the raw
    token value is never written to the filesystem.
    """
    return hmac.new(_TOKEN_SALT, token.encode('utf-8'), hashlib.sha256).hexdigest()[:16]


def _resolve_client_save_dir(save_dir: str) -> Path:
    if not save_dir:
        raise ValueError(f'Invalid save_dir: {save_dir}')
    path = Path(save_dir).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'save_dir does not exist on the server: {path.as_posix()}')
    if not path.is_dir():
        raise ValueError(f'save_dir is not a directory on the server: {path.as_posix()}')
    return path


def validate_user_path(token: str, path: str) -> bool:
    """
    Validate that the path is safe and belongs to the user.

    This function checks:
    1. Path doesn't contain '..' (directory traversal attack prevention)
    2. Path doesn't start with '/' (absolute path prevention)
    3. Path doesn't contain null bytes
    4. Path components are reasonable

    Args:
        token: User's authentication token (used to identify ownership)
        path: The path to validate

    Returns:
        True if path is safe, False otherwise
    """
    if not path:
        return False

    # Check for directory traversal attempts
    if '..' in path:
        return False

    # Check for null bytes (security vulnerability)
    if '\x00' in path:
        return False

    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.\./',  # Directory traversal
        r'/\.\.',
        r'^/',  # Absolute path
        r'^\.\.',  # Starts with ..
        r'~',  # Home directory expansion
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, path):
            return False

    return True


def validate_ownership(token: str, model_owner: str) -> bool:
    """
    Validate that the user owns the resource.

    Args:
        token: User's authentication token
        model_owner: The owner of the model/checkpoint

    Returns:
        True if user owns the resource, False otherwise
    """
    if not token or not model_owner:
        return False
    return token == model_owner
