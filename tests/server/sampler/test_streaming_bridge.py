# Copyright (c) ModelScope Contributors. All rights reserved.
"""Consumer disconnection never shortens backend ownership."""
from types import SimpleNamespace

import pytest

from twinkle.server.sampler.backends.streaming import STREAM_SENTINEL, stream_to_queue


@pytest.mark.parametrize('disconnect', [False, True])
def test_stream_drains_and_closes_after_consumer_disconnect(disconnect):
    consumed, closed, published = [], [], []

    def generate(*args):
        try:
            for index in range(3):
                consumed.append(index)
                yield index, None
        finally:
            closed.append(True)

    def put(value):
        if disconnect:
            raise RuntimeError('queue actor is gone')
        published.append(value)

    stream_to_queue(SimpleNamespace(sample_stream=generate), SimpleNamespace(put=put), [])
    assert consumed == [0, 1, 2] and closed == [True]
    assert published == ([] if disconnect else [(0, None), (1, None), (2, None), STREAM_SENTINEL])


def test_backend_failure_is_reported_and_iterator_closed():
    closed, published = [], []

    def generate(*args):
        try:
            yield 1, None
            raise ValueError('generation failed')
        finally:
            closed.append(True)

    stream_to_queue(SimpleNamespace(sample_stream=generate), SimpleNamespace(put=published.append), [])
    assert closed == [True]
    assert isinstance(published[-2], ValueError)
    assert published[-1] == STREAM_SENTINEL
