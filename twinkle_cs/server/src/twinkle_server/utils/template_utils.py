# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compatibility alias for :mod:`twinkle.template.mapping`."""

import sys

from twinkle.template import mapping as _implementation

sys.modules[__name__] = _implementation
