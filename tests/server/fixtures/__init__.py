# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared on-disk fixtures for server tests."""
from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent

# All-mock CPU-only server configs used by CLI tests and the in-process e2e
# Ray Serve startup test. The YAML is the abstraction — callers pass these
# paths to ServerConfig.from_yaml / CLI --config flags.
#
# - ``MOCK_SERVER_CONFIG``: file-backed persistence — no external deps,
#   safe in CI / locally without docker. The default.
# - ``MOCK_SERVER_CONFIG_REDIS``: redis-backed persistence — requires a
#   redis at ``redis://127.0.0.1:6379``. Use to exercise the redis state
#   backend through the same e2e flow.
MOCK_SERVER_CONFIG = FIXTURES_DIR / 'server_config_mock.yaml'
MOCK_SERVER_CONFIG_REDIS = FIXTURES_DIR / 'server_config_mock_redis.yaml'
