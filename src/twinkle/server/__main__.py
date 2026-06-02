# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI entry point for Twinkle Server.

Thin shim — delegates to the typer-based :mod:`twinkle.server.cli` so the
``python -m twinkle.server`` command and the ``twinkle-server`` console
script share one implementation.

Usage::

    python -m twinkle.server launch --config server_config.yaml
    python -m twinkle.server check-config --config server_config.yaml
    python -m twinkle.server print-config --config server_config.yaml
    python -m twinkle.server clear persistence --config server_config.yaml
"""
from __future__ import annotations

import sys

from twinkle.server.cli import main

if __name__ == '__main__':
    sys.exit(main())
