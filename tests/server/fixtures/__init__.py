# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared on-disk fixtures for server tests."""
from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent

# All-mock CPU-only server config used by CLI tests and the in-process e2e
# Ray Serve startup test. The YAML is the abstraction — callers pass this
# path to ServerConfig.from_yaml / CLI --config flags.
MOCK_SERVER_CONFIG = FIXTURES_DIR / 'server_config_mock.yaml'
