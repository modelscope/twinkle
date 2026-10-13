"""A batch owns generation until all of its prompts have exited."""
import pytest


@pytest.mark.asyncio
async def test_failed_prompt_waits_for_sibling_generation_to_exit():
    import asyncio
    from twinkle.data_format import SamplingParams
    from twinkle.sampler.vllm_sampler import vLLMSampler
    sampler = object.__new__(vLLMSampler)
    sampler.template = None
    sampler._run_in_loop = lambda coroutine: coroutine
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    async def single(feat, *args, **kwargs):
        if feat['input_ids'][0] == 1:
            raise ValueError('bad prompt')
        entered.set()
        await release.wait()
        finished.set()
        return object()
    sampler._sample_single = single
    generation = asyncio.create_task(sampler.sample(
        [{'input_ids': [1]}, {'input_ids': [2]}], SamplingParams(max_tokens=1)))
    try:
        await entered.wait()
        await asyncio.sleep(.01)
        assert not generation.done(), 'a failed prompt returned while its sibling was still generating'
        release.set()
        with pytest.raises(ValueError, match='bad prompt'):
            await generation
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(generation, return_exceptions=True)
