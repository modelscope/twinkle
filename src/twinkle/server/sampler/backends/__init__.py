# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler backend implementations.

The cross-process streaming bridge lives in ``streaming`` and is re-exported here
so a sibling backend module can import it from ``.streaming`` without importing
this package root, while external consumers keep using
``from twinkle.server.sampler.backends import stream_to_queue``.
"""
from .streaming import STREAM_SENTINEL, stream_to_queue

__all__ = ['STREAM_SENTINEL', 'stream_to_queue']
