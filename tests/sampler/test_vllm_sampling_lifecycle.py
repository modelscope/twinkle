"""Batch failure cancels siblings and waits for their actual cleanup."""
import asyncio

import pytest
from twinkle.sampler.vllm_sampler import vLLMSampler


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_batch', [False, True])
async def test_batch_waits_for_cancelled_sibling_cleanup(cancel_batch):
    entered = asyncio.Event()
    release = asyncio.Event()
    cleanup = asyncio.Event()
    finished = asyncio.Event()
    async def failed():
        await entered.wait()
        if cancel_batch:
            await asyncio.Future()
        raise ValueError('bad prompt')
    async def sibling():
        try:
            entered.set()
            await asyncio.Future()
        finally:
            cleanup.set()
            await release.wait()
            finished.set()
    generation = asyncio.create_task(vLLMSampler._gather_samples([failed(), sibling()]))
    try:
        await entered.wait()
        if cancel_batch:
            generation.cancel()
        await asyncio.wait_for(cleanup.wait(), 1)
        assert not generation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel_batch else ValueError):
            await generation
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(generation, return_exceptions=True)
