# Copyright (c) ModelScope Contributors. All rights reserved.
"""CS model head configuration contracts."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.server.integration.cs_tasks.task_models import _mcore_native_key, encoded_batch
from twinkle.server.config import ServerConfig
from twinkle.server.config.training import TrainingArguments
from twinkle.server.model.app import _validate_model_task

TASK_CONFIGS = Path(__file__).resolve().parents[1] / 'integration' / 'cs_tasks'


@pytest.mark.parametrize('options', [
    {'task_type': 'seq_cls'}, {'num_labels': 3}, {'task_type': 'causal_lm', 'num_labels': 1},
    {'task_type': 'seq_cls', 'num_labels': 0}, {'task_type': 'seq_cls', 'num_labels': True},
])
def test_task_head_arguments_fail_fast(options):
    with pytest.raises(ValidationError):
        TrainingArguments(**options)


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
def test_task_recipe_yaml_preserves_head_contract(backend):
    config = ServerConfig.from_yaml(TASK_CONFIGS / f'task_server_{backend}.yaml')
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


def test_cookbook_seq_cls_yaml_declares_heads():
    root = Path(__file__).resolve().parents[3]
    config = ServerConfig.from_yaml(root / 'cookbook/client/server/transformer/server_config_seq_cls.yaml')
    models = {model.name: model.config for model in config.training_config.models}
    assert models['Qwen3.5-4B-reranker'].num_labels == 1
    assert models['Qwen3.5-4B-seq-cls'].num_labels == 3
    assert all('score' in model.target_modules for model in models.values())


def test_mcore_native_weights_keep_the_language_model_prefix():
    assert _mcore_native_key('model.layers.0.self_attn.q_proj.weight') == \
        'model.language_model.layers.0.self_attn.q_proj.weight'
    assert _mcore_native_key('score.weight') == 'score.weight'
    assert _mcore_native_key('lm_head.weight') == 'lm_head.weight'


@pytest.mark.parametrize('task,width,architecture,actual,success', [
    ('seq_cls', 3, 'Qwen2ForSequenceClassification', 3, True),
    ('seq_cls', 1, 'Qwen2ForSequenceClassification', 3, False),
    # A causal backbone with a matching num_labels is the attached-head deployment: the
    # task rides the constructor (task='seq_cls'), no persisted head to validate.
    ('seq_cls', 3, 'Qwen2ForCausalLM', 3, True),
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
def test_task_batches_match_the_wire_contract(task):
    from pydantic import TypeAdapter
    from twinkle.protocol.types.data import WireInputBatch

    def tokenizer(text, **kwargs):
        return {'input_ids': [4, 5, 6], 'attention_mask': [1, 1, 1]}
    batch = encoded_batch(task, tokenizer)
    parsed = TypeAdapter(WireInputBatch).validate_python(batch)
    assert len(parsed) == 12
    assert all(type(label) is int for row in parsed for label in row.labels)


@pytest.mark.parametrize('task,width', [('seq_cls', 3), ('reranker', 1), ('value', 1)])
@pytest.mark.parametrize('mode', ['lora', 'full'])
def test_transformers_backend_preserves_validated_head_contract(monkeypatch, task, width, mode):
    from transformers import AutoConfig
    from twinkle.hub import HubOperation
    from twinkle.server.model import app
    from twinkle.server.model.backends import transformers_model

    monkeypatch.setattr(HubOperation, 'download_model', lambda path: path)
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(
        architectures=['Qwen2ForCausalLM']))
    received = {}
    model_class = 'TwinkleCompatFullTransformersModel' if mode == 'full' else 'TwinkleCompatTransformersModel'
    monkeypatch.setattr(transformers_model, model_class, lambda **kwargs: received.update(kwargs))
    kwargs = {'model_id': 'unused', 'task_type': task, 'num_labels': width, 'train_mode': mode}
    app._validate_model_task(kwargs)
    app._make_transformers_model(kwargs)
    assert received == {'model_id': 'unused', 'task': task, 'num_labels': width}


def test_full_persisted_head_does_not_forward_num_labels_to_hf(monkeypatch):
    from twinkle.server.model import app
    from twinkle.server.model.backends import transformers_model

    received = {}
    monkeypatch.setattr(transformers_model, 'TwinkleCompatFullTransformersModel',
                        lambda **kwargs: received.update(kwargs))
    app._make_transformers_model({'model_id': 'persisted-classifier', 'train_mode': 'full',
                                 'num_labels': 3, 'max_loras': 2})
    assert received == {'model_id': 'persisted-classifier'}


def test_full_model_attaches_head_and_preserves_device_mesh(monkeypatch):
    import inspect

    from twinkle import DeviceMesh
    from twinkle.infra.dispatch import _get_device_mesh_param_name
    from twinkle.server.model.backends import transformers_model

    mesh = DeviceMesh.from_sizes(dp_size=2, device_type='cpu')
    received = {}
    monkeypatch.setattr(transformers_model.TransformersModel, '__init__',
                        lambda self, **kwargs: received.update(kwargs))
    attached = []
    monkeypatch.setattr(transformers_model.TransformersModel, '_add_task_head',
                        lambda self, *args: attached.append(args))
    model_class = transformers_model.TwinkleCompatFullTransformersModel
    inspect.unwrap(model_class.__init__)(
        object.__new__(model_class), model_id='causal-backbone', device_mesh=mesh, task='seq_cls', num_labels=3)
    assert attached == [('', 'seq_cls', None, 3)]
    assert received['device_mesh'] is mesh
    assert _get_device_mesh_param_name(
        transformers_model.TwinkleCompatFullTransformersModel.__init__) == 'device_mesh'


def test_full_custom_model_class_uses_compat_dispatch(monkeypatch):
    from twinkle.server.model import app

    marker = type('CustomModel', (), {})
    received = {}
    monkeypatch.setattr(app, '_resolve_model_cls', lambda spec: marker)
    monkeypatch.setattr(app, '_make_model_class',
                        lambda base: lambda **kwargs: received.update(base=base, kwargs=kwargs))
    app._make_transformers_model({'model_id': 'st-model', 'train_mode': 'full',
                                  'model_cls': 'worker_only.encoder:Model'})
    assert received['base'] is marker
    assert received['kwargs'] == {'model_id': 'st-model'}


def test_embedding_deployment_accepts_explicit_task(monkeypatch):
    from transformers import AutoConfig

    from twinkle.hub import HubOperation

    args = TrainingArguments(task_type='embedding')
    monkeypatch.setattr(HubOperation, 'download_model', lambda path: path)
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(
        architectures=['Qwen2ForCausalLM']))
    kwargs = {'model_id': 'causal-backbone', 'task_type': args.task_type}
    _validate_model_task(kwargs)
    assert kwargs == {'model_id': 'causal-backbone'}


def test_decision_requires_a_family_head_factory():
    with pytest.raises(ValueError, match='family model_loader'):
        _validate_model_task({'task_type': 'decision', 'model_loader': SimpleNamespace()})
    with pytest.raises(ValidationError, match='model_loader'):
        TrainingArguments(task_type='decision')


@pytest.mark.parametrize('mode', ['lora', 'full'])
def test_transformers_decision_uses_loader_architecture(monkeypatch, mode):
    from twinkle.server.model import app
    from twinkle.server.model.backends import transformers_model

    received = {}
    for name in ('TwinkleCompatFullTransformersModel', 'TwinkleCompatTransformersModel'):
        monkeypatch.setattr(transformers_model, name, lambda **kw: received.update(kw))
    loader = SimpleNamespace(task_type='decision', task_head_loader=lambda: ('family', 'Loader'),
                             model_arch=SimpleNamespace(
        language_model=['model.language_model'], vision_tower=['model.visual'],
        aligner=['model.visual.merger'], lm_head='lm_head'))
    args = TrainingArguments(task_type='decision', model_loader='family:build')
    kwargs = {'model_id': 'decision-backbone', 'train_mode': mode,
              'task_type': args.task_type, 'model_loader': loader}
    app._validate_model_task(kwargs)
    app._make_transformers_model(kwargs)
    assert received['model_arch']['vision_tower'] == ['model.visual']
    if mode == 'full':
        assert 'task' not in received and 'task_head_loader' not in received
    else:
        assert received['task'] == 'decision'
        assert received['task_head_loader'] == ('family', 'Loader')


@pytest.mark.parametrize('model_cls', [None, 'swift.dev.model.sentence_transformer_model:SentenceTransformerModel'])
def test_native_st_guard_uses_resolved_directory(tmp_path, monkeypatch, model_cls):
    from transformers import AutoConfig
    from twinkle.hub import HubOperation

    (tmp_path / 'modules.json').write_text('[]')
    monkeypatch.setattr(HubOperation, 'download_model', lambda path: str(tmp_path))
    config_reads = []
    monkeypatch.setattr(AutoConfig, 'from_pretrained',
                        lambda *a, **kw: config_reads.append(a) or SimpleNamespace(architectures=['BertModel']))
    kwargs = {'model_id': 'ms://sentence-transformers/LaBSE', 'task_type': 'embedding'}
    if model_cls is not None:
        kwargs['model_cls'] = model_cls
        _validate_model_task(kwargs)
        assert config_reads
    else:
        with pytest.raises(ValueError, match='native sentence-transformers checkpoint requires model_cls'):
            _validate_model_task(kwargs)
        assert not config_reads
