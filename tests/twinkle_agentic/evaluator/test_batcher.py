from concurrent.futures import ThreadPoolExecutor

import pytest

from twinkle.data_format import SamplingParams
from twinkle_agentic.evaluator._batcher import SamplerBatcher

from .conftest import RecordingSampler


def _trajectory(value):
    return {'messages': [{'role': 'user', 'content': value}]}


def test_compatible_requests_are_batched_and_tail_is_padded():
    sampler = RecordingSampler(dp_world_size=4)
    batcher = SamplerBatcher(sampler, batch_size=4, batch_wait_ms=20, sampler_kwargs={})
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            result = list(pool.map(lambda i: batcher.submit(_trajectory(str(i)), SamplingParams(max_tokens=3)), range(3)))
        assert len(result) == 3
        assert len(sampler.calls) == 1
        assert len(sampler.calls[0][0]) == 4
    finally:
        batcher.close()
    assert not batcher._worker.is_alive()


def test_incompatible_requests_do_not_share_a_sampler_call():
    sampler = RecordingSampler()
    batcher = SamplerBatcher(sampler, batch_size=2, batch_wait_ms=0, sampler_kwargs={})
    try:
        batcher.submit(_trajectory('a'), SamplingParams(max_tokens=1))
        batcher.submit(_trajectory('b'), SamplingParams(max_tokens=2))
        assert [call[1].max_tokens for call in sampler.calls] == [1, 2]
    finally:
        batcher.close()


def test_sampler_error_is_delivered_to_all_requests():
    sampler = RecordingSampler()
    sampler.sample = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('boom'))
    batcher = SamplerBatcher(sampler, batch_size=2, batch_wait_ms=20, sampler_kwargs={})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(batcher.submit, _trajectory(str(i)), SamplingParams()) for i in range(2)]
            for future in futures:
                with pytest.raises(RuntimeError, match='boom'):
                    future.result()
    finally:
        batcher.close()
