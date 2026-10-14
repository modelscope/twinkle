"""Token-only sampling, batch cleanup and checkpoint write boundaries."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from safetensors.torch import save_file

from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.exceptions import FullWeightLoadError
from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler


@pytest.mark.parametrize('kind', ['missing', 'empty', 'bad_index', 'missing_shard', 'corrupt_shard'])
def test_invalid_checkpoint_does_not_write_weights(tmp_path, kind):
    path = tmp_path / 'checkpoint'
    if kind != 'missing':
        path.mkdir()
    if kind == 'bad_index':
        (path / 'model.safetensors.index.json').write_text('{')
    if kind == 'missing_shard':
        (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'absent.safetensors'}}))
    if kind == 'corrupt_shard':
        (path / 'model.safetensors').write_text('not a checkpoint')
    sampler = object.__new__(vLLMSampler)
    # This existing local directory avoids a hub request for the missing case.
    from unittest.mock import patch
    writes = []
    sampler.model_id = str(path)
    sampler.engine = SimpleNamespace(update_weights=lambda *a, **k: writes.append(True))
    with patch('twinkle.sampler.vllm_sampler.vllm_sampler.HubOperation.download_model', return_value=str(path)):
        with pytest.raises(Exception) as error:
            sampler.load_full_weights_from_path(str(path))
    assert not isinstance(error.value, FullWeightLoadError)
    assert writes == []
    assert not hasattr(sampler, '_loaded_full_weights_path')


def test_partial_write_is_reported_as_weight_failure(tmp_path):
    save_file({'weight': torch.ones(1)}, str(tmp_path / 'model.safetensors'))
    sampler = object.__new__(vLLMSampler)
    writes = []
    async def update(weights, **kwargs):
        writes.extend(list(weights))
        raise RuntimeError('engine failed after writing')
    sampler.engine = SimpleNamespace(update_weights=update)
    sampler._run_in_loop = asyncio.run
    with pytest.raises(FullWeightLoadError) as error:
        sampler.load_full_weights_from_path(str(tmp_path))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert len(writes) == 1
    assert not hasattr(sampler, '_loaded_full_weights_path')


class _OtherClientTemplate:
    def get_vllm_input_ids(self, *args):
        raise AssertionError('Token-only sampling rewrote encoded input IDs')

    def concat_input_feature(self, *args):
        raise AssertionError('Token-only sampling built unused training features')

    def decode(self, *args):
        raise AssertionError('Token-only sampling decoded unused text')


@pytest.mark.asyncio
@pytest.mark.parametrize('template', [None, _OtherClientTemplate()])
@pytest.mark.parametrize('logprobs_only', [False, True])
async def test_token_only_sampling_needs_no_template(template, logprobs_only):
    sampler = object.__new__(vLLMSampler)
    sampler.template = template
    sampler.engine = SimpleNamespace(sample=AsyncMock(return_value=SampleResponse(
        prompt_token_ids=[7],
        sequences=[SampledSequence(tokens=[8], logprobs=[[(8, -.1)]], stop_reason='length')],
        prompt_logprobs=[-.2],
    )))
    result = await sampler._sample_single(
        {'input_ids': [7]}, SamplingParams(max_tokens=1), token_only=True, logprobs_only=logprobs_only)
    assert sampler.engine.sample.call_args.kwargs['prompt'] == [7]
    assert result.prompt_token_ids == [7] and result.prompt_logprobs == [-.2]
    assert result.sequences[0].tokens == ([] if logprobs_only else [8])
    assert result.sequences[0].logprobs == (None if logprobs_only else [[(8, -.1)]])
    assert result.sequences[0].decoded is None and result.sequences[0].new_input_feature is None


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
