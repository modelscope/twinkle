# Copyright (c) ModelScope Contributors. All rights reserved.
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("twinkle-server")
except PackageNotFoundError:
    __version__ = "0.4.0.dev0"
