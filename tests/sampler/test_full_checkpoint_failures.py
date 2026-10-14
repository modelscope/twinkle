"""Validate checkpoints before writing; distinguish partial engine writes."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

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
