"""Packed boundary metadata and model patch installation are separate steps."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor


def _config(model_type='qwen3_5', attention='flash_attention_2'):
    return SimpleNamespace(model_type=model_type, _attn_implementation=attention)


@pytest.mark.parametrize('padding_free', [False, True])
@pytest.mark.parametrize('model_type', ['qwen3_5', 'qwen3_5_moe'])
def test_packed_boundaries_without_model_argument(padding_free, model_type):
    processor = InputProcessor(padding_free=padding_free)
    features = [{'position_ids': torch.tensor([[0, 1, 2, 0, 1]])}]
    out = processor.prepare_transformers_packed_args(features, hf_config=_config(model_type))
    assert out[0]['cu_seq_lens_q'].tolist() == [0, 3, 5]
    assert out[0]['cu_seq_lens_k'].tolist() == [0, 3, 5]
    assert out[0]['max_length_q'] == 3
    assert out[0]['max_length_k'] == 3
    assert out[0]['cu_seq_lens_q'].dtype == torch.int32


def test_model_patch_is_installed_on_owner_once(monkeypatch):
    owner = SimpleNamespace(model=SimpleNamespace(), hf_config=_config(), _enable_sp=False)
    calls = []

    def apply_patch(model, patch_cls, **kwargs):
        calls.append((model, kwargs))
        model._twinkle_gdn_padding_free_patched = True

    monkeypatch.setattr('twinkle.model.transformers.transformers.apply_patch', apply_patch)
    inputs = {'cu_seq_lens_q': torch.tensor([0, 3, 5])}
    TransformersModel._ensure_gdn_padding_free_patch(owner, inputs)
    TransformersModel._ensure_gdn_padding_free_patch(owner, inputs)
    assert calls == [(owner.model, {'hf_config': owner.hf_config, 'enable_sp': False})]


@pytest.mark.parametrize('enable_sp,inputs', [(True, {'cu_seq_lens_q': object()}), (False, {})])
def test_sp_and_normal_batches_do_not_install_gdn_patch(monkeypatch, enable_sp, inputs):
    patch = Mock()
    monkeypatch.setattr('twinkle.model.transformers.transformers.apply_patch', patch)
    owner = SimpleNamespace(model=SimpleNamespace(), hf_config=_config(), _enable_sp=enable_sp)
    TransformersModel._ensure_gdn_padding_free_patch(owner, inputs)
    patch.assert_not_called()


def test_sp_metadata_path_is_unchanged():
    processor = InputProcessor(padding_free=True)
    features = [{'position_ids': torch.tensor([[0, 1, 0, 1]])}]
    out = processor.prepare_transformers_packed_args(features, hf_config=_config(), enable_sp=True)
    assert 'cu_seq_lens_q' not in out[0]


def test_other_model_families_do_not_receive_gdn_metadata():
    processor = InputProcessor(padding_free=True)
    features = [{'position_ids': torch.tensor([[0, 1, 0, 1]])}]
    out = processor.prepare_transformers_packed_args(features, hf_config=_config('llama'))
    assert 'cu_seq_lens_q' not in out[0]


def test_packed_batch_still_requires_flash_attention():
    processor = InputProcessor(padding_free=True)
    with pytest.raises(RuntimeError, match='FlashAttention'):
        processor.prepare_transformers_packed_args(
            [{'position_ids': torch.tensor([[0, 1, 0, 1]])}], hf_config=_config(attention='sdpa'))


def test_processor_pipeline_emits_metadata_without_model():
    processor = InputProcessor(padding_free=True)
    batch = [{'input_ids': [1, 2, 3], 'position_ids': [0, 1, 2], 'labels': [1, 2, 3]},
             {'input_ids': [4, 5], 'position_ids': [0, 1], 'labels': [4, 5]}]
    out = processor(batch, hf_config=_config())
    assert out['cu_seq_lens_q'].tolist() == [0, 3, 5]
    assert out['input_ids'].shape == (1, 5)
