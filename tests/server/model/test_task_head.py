# Copyright (c) ModelScope Contributors. All rights reserved.
"""CS model head configuration contracts."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cookbook.client.twinkle.prepare_sequence_model import _mcore_native_key
from twinkle.server.config import ServerConfig
from twinkle.server.config.training import TrainingArguments
from twinkle.server.model.app import _validate_model_task


@pytest.mark.parametrize('options', [
    {'task_type': 'seq_cls'}, {'num_labels': 3}, {'task_type': 'causal_lm', 'num_labels': 1},
    {'task_type': 'seq_cls', 'num_labels': 0}, {'task_type': 'seq_cls', 'num_labels': True},
])
def test_task_head_arguments_fail_fast(options):
    with pytest.raises(ValidationError):
        TrainingArguments(**options)


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
def test_task_recipe_yaml_preserves_head_contract(backend):
    root = Path(__file__).resolve().parents[3]
    config = ServerConfig.from_yaml(root / f'cookbook/client/twinkle/task_server_{backend}.yaml')
    models = {model.name: model for model in config.training_config.models}
    assert set(models) == {'cs-embedding', 'cs-reranker', 'cs-seq_cls'}
    assert models['cs-reranker'].config.num_labels == 1
    assert models['cs-seq_cls'].config.num_labels == 3
    assert models['cs-embedding'].config.num_labels is None
    head = 'score' if backend == 'transformers' else 'output_layer'
    assert head in models['cs-reranker'].config.target_modules
    assert models['cs-reranker'].capacity == 2
    if backend == 'megatron':
        app = next(app for app in config.applications if app.name == 'reranker')
        assert app.args.language_model_only is True


def test_qwen35_megatron_yaml_uses_native_bases():
    root = Path(__file__).resolve().parents[3]
    config = ServerConfig.from_yaml(root / 'cookbook/client/twinkle/task_server_megatron_qwen35.yaml')
    apps = [app for app in config.applications if app.import_path == 'model']
    assert len(apps) == 3
    for app in apps:
        assert app.args.model_id.startswith('output/cs-models-mcore/')
        assert app.args.language_model_only is True


def test_mcore_native_weights_keep_the_language_model_prefix():
    assert _mcore_native_key('model.layers.0.self_attn.q_proj.weight') == \
        'model.language_model.layers.0.self_attn.q_proj.weight'
    assert _mcore_native_key('score.weight') == 'score.weight'
    assert _mcore_native_key('lm_head.weight') == 'lm_head.weight'


@pytest.mark.parametrize('task,width,architecture,actual,success', [
    ('seq_cls', 3, 'Qwen2ForSequenceClassification', 3, True),
    ('seq_cls', 1, 'Qwen2ForSequenceClassification', 3, False),
    ('seq_cls', 3, 'Qwen2ForCausalLM', 3, False),
    ('causal_lm', None, 'Qwen2ForSequenceClassification', 3, False),
    ('causal_lm', None, 'Qwen2ForCausalLM', None, True),
])
def test_persisted_head_matches_deployment(tmp_path, monkeypatch, task, width, architecture, actual, success):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig
    save_file({'score.weight': torch.zeros(actual or 1, 8)}, str(tmp_path / 'model.safetensors'))
    from twinkle.hub import HubOperation
    monkeypatch.setattr(HubOperation, 'download_model', lambda path: path)
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(
        architectures=[architecture], num_labels=actual))
    kwargs = {'model_id': str(tmp_path), 'task_type': task, 'num_labels': width}
    if success:
        _validate_model_task(kwargs)
    else:
        with pytest.raises(ValueError):
            _validate_model_task(kwargs)


def test_default_model_path_does_not_load_head_metadata(monkeypatch):
    from twinkle.hub import HubOperation

    def unexpected(*args, **kwargs):
        raise AssertionError('Default deployments must keep their existing loading path')
    monkeypatch.setattr(HubOperation, 'download_model', unexpected)
    _validate_model_task({'model_id': 'unused'})


@pytest.mark.parametrize('task', ['embedding', 'reranker', 'seq_cls'])
def test_cookbook_batches_match_the_wire_contract(task):
    import importlib
    from pydantic import TypeAdapter
    from twinkle.protocol.types.data import WireInputBatch

    recipe = importlib.import_module('cookbook.client.twinkle.' + task)
    def tokenizer(text, **kwargs):
        return {'input_ids': [4, 5, 6], 'attention_mask': [1, 1, 1]}
    batch = recipe.encoded_batch(tokenizer)
    parsed = TypeAdapter(WireInputBatch).validate_python(batch)
    assert len(parsed) == 12
    assert all(type(label) is int for row in parsed for label in row.labels)
