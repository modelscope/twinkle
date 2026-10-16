from types import SimpleNamespace
from unittest.mock import patch

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
