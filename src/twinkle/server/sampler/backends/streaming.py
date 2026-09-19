# Copyright (c) ModelScope Contributors. All rights reserved.
"""Cross-process streaming bridge for sampler backends.

Lives in its own module (rather than the ``backends`` package ``__init__``) so
that a backend implementation (e.g. ``mock_sampler``) can import it without
importing its own package root — which would be a package-initialisation-order
dependency. ``backends/__init__`` re-exports these names, so external consumers
(``from twinkle.server.sampler.backends import stream_to_queue``) are unchanged.
"""
from __future__ import annotations

STREAM_SENTINEL = '__STREAM_END__'


def stream_to_queue(sampler, queue, inputs, sampling_params=None, adapter_name='', adapter_path=None):
    """Push streaming deltas from *sampler* to a cross-process Ray queue.

    Works with any object that exposes a ``sample_stream`` iterator.
    """
    try:
        for delta, reason in sampler.sample_stream(inputs, sampling_params, adapter_name, adapter_path):
            queue.put((delta, reason))
    except Exception as e:
        queue.put(e)
    finally:
        queue.put(STREAM_SENTINEL)
