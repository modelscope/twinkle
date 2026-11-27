# Copyright (c) ModelScope Contributors. All rights reserved.
# The version is never written here: every distribution reads the single VERSION file at the
# repository root (see twinkle_cs/README.md).
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("twinkle-server")
except PackageNotFoundError:
    # Source checkout without an installed distribution.
    __version__ = (Path(__file__).resolve().parents[4] / "VERSION").read_text().strip()
