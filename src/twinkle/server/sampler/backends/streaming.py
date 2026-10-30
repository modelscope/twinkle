# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lightweight queue bridge shared by real and mock sampling backends.

Kept independent of the HTTP runtime so backend workers do not import FastAPI
or depend on the mock backend. Both consumers import this defining module.
"""
from __future__ import annotations

STREAM_SENTINEL = '__STREAM_END__'


class StreamPublisher:
    """A disconnected consumer cannot end the backend generation lifetime."""

    def __init__(self, queue):
        self.queue = queue
        self.connected = True

    def put(self, item):
        if not self.connected:
            return
        try:
            self.queue.put(item)
        except Exception:
            # Ray queue transport can disappear after the HTTP reader closes.
            self.connected = False


def stream_to_queue(sampler, queue, inputs, sampling_params=None, adapter_name='', adapter_path=None):
    """Drain and close the generator even when its HTTP consumer disconnects."""
    from contextlib import ExitStack

    publisher = StreamPublisher(queue)
    generation = sampler.sample_stream(inputs, sampling_params, adapter_name, adapter_path)
    with ExitStack() as cleanup:
        cleanup.callback(publisher.put, STREAM_SENTINEL)
        close = getattr(generation, 'close', None)
        if close is not None:
            cleanup.callback(close)
        try:
            for delta, reason in generation:
                publisher.put((delta, reason))
        except Exception as error:
            publisher.put(error)
