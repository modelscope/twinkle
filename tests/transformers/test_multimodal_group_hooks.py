"""Exercise real Qwen3.5 template hooks with multiple optimizer-backed groups."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import ProcessorMixin

from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template.qwen3_5_vl import Qwen3_5Template


class Visual(torch.nn.Module):
    dtype = torch.float32

    def forward(self, values, grid_thw):
        return values


class VisionLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 3)
        self.model.visual = Visual()
        self.lm_head = torch.nn.Linear(3, 4)
        self.config = SimpleNamespace(image_token_id=0, video_token_id=3)
        self.fail = False

    def forward(self, inputs_embeds, position_ids=None):
        if self.fail:
            raise RuntimeError('model forward failed')
        return {'logits': self.lm_head(inputs_embeds)}


class EncodedProcessor(InputProcessor):
    def __call__(self, inputs, **kwargs):
        return dict(inputs)


@pytest.fixture
def wrapper():
    model = VisionLanguageModel()
    wrapper = object.__new__(TransformersModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.active_group = 'b'
    wrapper.device_mesh = None
    wrapper.hf_config = model.config
    wrapper.sp_strategy = None
    wrapper._model_wrapped = False
    wrapper._enable_expert_parallel = False
    wrapper._gdn_padding_free_enabled = False
    wrapper._maybe_apply_expert_parallel = lambda: None
    wrapper._ensure_sp_strategy = lambda: None
    wrapper._ensure_gdn_padding_free_patch = lambda: None
    wrapper.strategy = SimpleNamespace(
        capture_pre_ep_state_if_needed=lambda *args, **kwargs: None,
        wrap_model=lambda model, *optimizers: (model, *optimizers),
        unwrap_model=lambda model: model,
    )
    groups = {}
    for name, parameter in zip(('a', 'b'), model.lm_head.parameters()):
        template = object.__new__(Qwen3_5Template)
        template.processor = object.__new__(ProcessorMixin)
        template.pre_forward_hook = Mock(wraps=template.pre_forward_hook)
        groups[name] = SimpleNamespace(
            optimizer=torch.optim.Adam([parameter]), template=template, processor=EncodedProcessor(),
            loss_instance=SimpleNamespace(require_logps=False), train_status=SimpleNamespace(),
            eval_status=SimpleNamespace(), accumulate_metrics=lambda training: None, _handler=None,
        )
    wrapper.optimizer_group = groups
    return wrapper


def inputs():
    return {'input_ids': torch.tensor([[0, 1]]), 'pixel_values': torch.tensor([[1., 2., 3.]]),
            'image_grid_thw': torch.tensor([[1, 1, 1]])}


@pytest.mark.parametrize('method', ['forward', 'forward_only'])
@pytest.mark.parametrize('first', ['a', 'b'])
def test_only_requested_multimodal_group_processes_each_forward(wrapper, method, first):
    expected_calls = {'a': 0, 'b': 0}
    for name in (first, 'b' if first == 'a' else 'a', first):
        result = getattr(wrapper, method)(inputs=inputs(), adapter_name=name, return_logits=True)
        assert result['logits'].shape == (1, 2, 4)
        assert result['logits'].requires_grad == (method == 'forward')
        expected_calls[name] += 1
        for identity, group in wrapper.optimizer_group.items():
            assert group.template.pre_forward_hook.call_count == expected_calls[identity]
        assert wrapper.active_group == 'b'


@pytest.mark.parametrize('method', ['forward', 'forward_only'])
def test_failed_forward_does_not_leave_another_groups_template_selected(wrapper, method):
    wrapper.model.fail = True
    with pytest.raises(RuntimeError, match='model forward failed'):
        getattr(wrapper, method)(inputs=inputs(), adapter_name='a')
    wrapper.model.fail = False
    assert wrapper.model(**inputs())['logits'].shape == (1, 2, 4)
    assert wrapper.optimizer_group['a'].template.pre_forward_hook.call_count == 1
    assert wrapper.optimizer_group['b'].template.pre_forward_hook.call_count == 1
    result = getattr(wrapper, method)(inputs=inputs(), return_logits=True)
    assert result['logits'].shape == (1, 2, 4)
    assert wrapper.optimizer_group['a'].template.pre_forward_hook.call_count == 1
    assert wrapper.optimizer_group['b'].template.pre_forward_hook.call_count == 2
