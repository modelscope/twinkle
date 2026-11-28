# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI entry point for Twinkle Server.

Thin shim — delegates to the typer-based :mod:`twinkle_server.cli` so the
``python -m twinkle_server`` command and the ``twinkle-server`` console
script share one implementation.

Usage::

    python -m twinkle_server launch --config server_config.yaml
    python -m twinkle_server check-config --config server_config.yaml
    python -m twinkle_server print-config --config server_config.yaml
    python -m twinkle_server clear persistence --config server_config.yaml
"""
from __future__ import annotations

import sys

from twinkle_server.cli import main

if __name__ == '__main__':
    sys.exit(main())
