import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from twinkle.model.transformers.transformers import TransformersModel


def test_resident_reset_removes_previous_templates_forward_hook():
    class ScalingTemplate:
        def __init__(self, scale):
            self.scale = scale

        def pre_forward_hook(self, model, args, kwargs):
            return (args[0] * self.scale,), kwargs

    def group(scale):
        return SimpleNamespace(template=ScalingTemplate(scale), _handler=None)

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    wrapper = object.__new__(TransformersModel)
    wrapper.__dict__['model'] = model
    wrapper.model_id = 'local-reset-test'
    wrapper._model_wrapped = True
    wrapper.strategy = SimpleNamespace(unwrap_model=lambda value: value,
                                      load_full_state_dict=lambda value, state: value.load_state_dict(state))
    wrapper.optimizer_group = {'': group(3)}
    object.__setattr__(wrapper, '_construct_default_optimizer_group', lambda: group(1))
    wrapper.register_mm_forward_hook(wrapper.optimizer_group[''])
    assert model(torch.ones(1, 1)).item() == 3
    initial = {'weight': torch.ones(1, 1)}
    with patch('twinkle.model.transformers.transformers._read_hf_state_dict', return_value=initial):
        for _ in range(2):
            model.weight.grad = torch.ones_like(model.weight)
            wrapper.reload_initial_weights(reset_training_state=True)
            assert model(torch.ones(1, 1)).item() == 1
            assert model.weight.grad is None
            assert len(model._forward_pre_hooks) == 1


def test_blank_model_still_resets_training_state():
    wrapper = object.__new__(TransformersModel)
    wrapper.model_id = ''
    wrapper._model_wrapped = False
    previous = SimpleNamespace(template=None, _handler=None, optimizer=object())
    fresh = SimpleNamespace(template=None, _handler=None, optimizer=None)
    wrapper.optimizer_group = {'': previous}
    object.__setattr__(wrapper, '_construct_default_optimizer_group', lambda: fresh)
    wrapper.reload_initial_weights(reset_training_state=True)
    assert wrapper.optimizer_group == {'': fresh}
    assert wrapper.active_group == '' and fresh.adapter_name == ''


@pytest.mark.parametrize('reload_kwargs', [{}, {'reset_training_state': False}], ids=['default', 'explicit-false'])
def test_transformers_reload_preserves_training_progress_by_default(reload_kwargs):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(3)
    model.weight.grad = torch.ones_like(model.weight)
    previous = SimpleNamespace(optimizer=object(), cur_step=7)
    wrapper = object.__new__(TransformersModel)
    wrapper.__dict__['model'] = model
    wrapper.model_id = 'local-reset-test'
    wrapper.optimizer_group = {'': previous}
    wrapper.strategy = SimpleNamespace(load_full_state_dict=lambda value, state: value.load_state_dict(state))
    initial = {'weight': torch.ones_like(model.weight)}
    with patch('twinkle.model.transformers.transformers._read_hf_state_dict', return_value=initial):
        wrapper.reload_initial_weights(**reload_kwargs)
    torch.testing.assert_close(model.weight, initial['weight'])
    assert model.weight.grad is None
    assert wrapper.optimizer_group[''] is previous and previous.cur_step == 7


def test_transformers_reload_propagates_checkpoint_failure():
    wrapper = object.__new__(TransformersModel)
    wrapper.model_id = 'unreadable-checkpoint'
    with patch('twinkle.model.transformers.transformers._read_hf_state_dict',
               side_effect=OSError('checkpoint unavailable')):
        with pytest.raises(OSError, match='checkpoint unavailable'):
            wrapper.reload_initial_weights()


@pytest.mark.parametrize('reload_kwargs', [{}, {'reset_training_state': False}, {'reset_training_state': True}],
                         ids=['default', 'explicit-false', 'reset'])
def test_megatron_reload_resets_optimizer_state_only_when_requested(reload_kwargs):
    # Exercise the repository method without importing the optional Megatron stack.
    path = Path('src/twinkle/model/megatron/megatron.py')
    model_class = next(node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == 'MegatronModel')
    method = next(node for node in model_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'reload_initial_weights')
    method.decorator_list = []
    namespace = {'_default_adapter_name': '', 'dist': SimpleNamespace(is_initialized=lambda: False)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    original_weights = {'weight': torch.ones_like(model.weight)}

    def load_weights(chunks, checkpoint, *, peft_format):
        assert checkpoint == 'initial-checkpoint' and peft_format is False
        for chunk in chunks:
            chunk.load_state_dict(original_weights)

    previous = SimpleNamespace(optimizer=object(), cur_step=7)
    fresh = SimpleNamespace(optimizer=None, cur_step=0)
    wrapper = SimpleNamespace(
        model=[model], _model_path='initial-checkpoint',
        optimizer_group={'': previous},
        strategy=SimpleNamespace(unwrap_model=lambda value: value,
                                 bridge=SimpleNamespace(load_weights=load_weights)),
        _is_model_ddp_wrapped=lambda: False,
        _construct_default_optimizer_group=lambda: fresh,
    )
    namespace['reload_initial_weights'](wrapper, **reload_kwargs)
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, original_weights['weight'])
    assert wrapper.optimizer_group[''] is (fresh if reload_kwargs.get('reset_training_state') else previous)
    if reload_kwargs.get('reset_training_state'):
        assert fresh.adapter_name == '' and fresh.optimizer is None and fresh.cur_step == 0


def _megatron_dpo_loss_callback(*, forward_only, disable_lora, kwargs=None, ref_logps=None, return_logits=False):
    from twinkle.loss import DPOLoss

    path = Path(__file__).resolve().parents[2] / 'src/twinkle/model/megatron/megatron.py'
    model_class = next(node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == 'MegatronModel')
    method = next(node for node in model_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'forward_backward')
    compute_loss = next(node for node in method.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'compute_loss' for target in node.targets))
    callback = next(node for node in method.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'post_loss_function')
    namespace = {
        'torch': torch, 'ModelOutput': dict,
        'Platform': SimpleNamespace(get_local_device=lambda: 'cpu'),
        'forward_only': forward_only, 'disable_lora': disable_lora, 'kwargs': dict(kwargs or {}),
        '_mb_counter': [0], 'return_logits': return_logits, 'loss_instance': DPOLoss(),
        'loss_extra_kwargs_per_mb': [{} if ref_logps is None else {'ref_logps': ref_logps}],
        'self': SimpleNamespace(strategy=SimpleNamespace(
            reduce_loss=lambda loss, count, logits, logps: [loss, count, {'logits': logits, 'logps': logps}])),
    }
    exec(compile(ast.Module(body=[compute_loss, callback], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['post_loss_function']


def test_megatron_adapter_disabled_reference_does_not_require_dpo_references():
    callback = _megatron_dpo_loss_callback(forward_only=True, disable_lora=True)
    logps = torch.tensor([[-1., -2.], [-3., -4.]])
    result = callback(torch.zeros(2, 2, 4), {'labels': torch.ones(2, 2, dtype=torch.long)}, logps)
    assert result[0].item() == 0 and result[1].item() == 0
    assert result[2]['logps'] is logps and result[2]['logits'] is None


@pytest.mark.parametrize('forward_only,disable_lora,kwargs', [
    (False, False, {}), (False, True, {}), (True, False, {}), (True, True, {'compute_loss': True}),
])
def test_megatron_training_and_explicit_loss_probes_still_require_dpo_references(
        forward_only, disable_lora, kwargs):
    callback = _megatron_dpo_loss_callback(forward_only=forward_only, disable_lora=disable_lora, kwargs=kwargs)
    with pytest.raises(ValueError, match='requires ref_logps'):
        callback(torch.zeros(2, 2, 4), {'labels': torch.ones(2, 2, dtype=torch.long)},
                 torch.tensor([[-1., -2.], [-3., -4.]]))


@pytest.mark.parametrize('forward_only', [False, True])
def test_megatron_non_reference_dpo_still_computes_a_nonzero_loss(forward_only):
    callback = _megatron_dpo_loss_callback(forward_only=forward_only, disable_lora=False,
                                         ref_logps=torch.zeros(2, 2))
    result = callback(torch.zeros(2, 2, 4), {'labels': torch.ones(2, 2, dtype=torch.long)},
                      torch.tensor([[-1., -2.], [-3., -4.]]))
    assert result[0].item() > 0 and result[1].item() == 1


def test_megatron_empty_probe_reports_pooled_scores_not_token_logits():
    callback = _megatron_dpo_loss_callback(forward_only=True, disable_lora=True, return_logits=True)
    pooled = torch.randn(2, 3)
    result = callback(torch.randn(2, 7, 3), {}, None, unpacked_logits=pooled)
    torch.testing.assert_close(result[2]['logits'], pooled)
    assert result[0].item() == 0 and result[1].item() == 0


def test_megatron_empty_probe_reports_canonical_embeddings():
    callback = _megatron_dpo_loss_callback(forward_only=True, disable_lora=True)
    embeddings = torch.randn(2, 8, requires_grad=True)
    result = callback(embeddings, {}, None, embeddings=embeddings)
    torch.testing.assert_close(result[2]['embeddings'], embeddings)
    assert not result[2]['embeddings'].requires_grad


def test_megatron_empty_logits_are_omitted_before_dp_collection():
    path = Path(__file__).resolve().parents[2] / 'src/twinkle/model/megatron/megatron.py'
    model_class = next(node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == 'MegatronModel')
    method = next(node for node in model_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'forward_backward')
    normalizer = next(node for node in reversed(method.body) if isinstance(node, ast.If)
                      and any(isinstance(child, ast.Name) and child.id == 'return_logits'
                              for child in ast.walk(node.test)))
    namespace = {'logits': [], 'return_logits': True, 'torch': torch}
    exec(compile(ast.Module(body=[normalizer], type_ignores=[]), str(path), 'exec'), namespace)
    assert namespace['logits'] is None

    logits = torch.randn(2, 3)
    namespace['logits'] = logits
    exec(compile(ast.Module(body=[normalizer], type_ignores=[]), str(path), 'exec'), namespace)
    assert namespace['logits'] is logits


def test_megatron_seq_cls_lora_checkpoint_opens_bridge_head_load_gate():
    from twinkle.model.megatron.multi_lora_megatron import _seq_cls_head_lora_load_gate

    class Bridge:
        _peft_format = True
        config = SimpleNamespace(task_type='seq_cls')
        hf_score_key = 'score.weight'

        def __init__(self):
            self.loaded_keys = None

        def _convert_post_process(self, mg_model, hf_state_dict, hf_prefix, to_mcore):
            self.loaded_keys = set(hf_state_dict)

    bridge = Bridge()
    checkpoint = {'score.lora_A.weight': object(), 'score.lora_B.weight': object()}
    with _seq_cls_head_lora_load_gate(bridge):
        bridge._convert_post_process(None, checkpoint, '', True)
    assert bridge.loaded_keys == {'score.weight', 'score.lora_A.weight', 'score.lora_B.weight'}
    assert 'score.weight' not in checkpoint
    assert bridge._convert_post_process.__self__ is bridge
