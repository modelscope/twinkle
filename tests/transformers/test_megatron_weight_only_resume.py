from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from twinkle.model.megatron.megatron import MegatronModel
from twinkle.model.megatron.multi_lora_megatron import MultiLoraMegatronModel


class MasterParameterOptimizer:
    """Use real Adam with the documented MCore model/master parameter contract."""
    def __init__(self, model_parameter):
        self.model_parameter = model_parameter
        self.master = torch.nn.Parameter(model_parameter.detach().float().clone())
        self.adam = torch.optim.Adam([self.master], lr=0.1)

    def reload_model_params(self):
        self.master.data.copy_(self.model_parameter)

    def step(self):
        self.master.grad = torch.ones_like(self.master)
        self.adam.step()
        self.model_parameter.data.copy_(self.master.to(self.model_parameter.dtype))


@pytest.mark.parametrize('model_class', [MegatronModel, MultiLoraMegatronModel])
def test_weights_only_load_refreshes_target_master_params_and_preserves_adam_state(tmp_path, model_class):
    parameter = torch.nn.Parameter(torch.tensor([7.], dtype=torch.bfloat16))
    optimizer = MasterParameterOptimizer(parameter)
    optimizer.step()
    before = deepcopy(optimizer.adam.state_dict())
    expected = MasterParameterOptimizer(torch.nn.Parameter(torch.tensor([2.], dtype=torch.bfloat16)))
    expected.adam.load_state_dict(deepcopy(before))
    expected.step()
    other = MasterParameterOptimizer(torch.nn.Parameter(torch.tensor([11.], dtype=torch.bfloat16)))
    group = SimpleNamespace(optimizer=optimizer)

    def load_weights(*args, **kwargs):
        parameter.data.fill_(2.)

    wrapper = SimpleNamespace(
        _get_default_group=lambda: 'target', model=[parameter],
        strategy=SimpleNamespace(unwrap_model=lambda model: model,
                                 bridge=SimpleNamespace(load_weights=load_weights)),
        optimizer_group={'target': group, 'other': SimpleNamespace(optimizer=other)},
        multi_adapter=SimpleNamespace(save_context=lambda adapter: nullcontext('slot'), load_lora_converter=None),
    )
    model_class.load(wrapper, str(tmp_path), adapter_name='target', load_optimizer=False)
    assert parameter.item() == 2.
    assert optimizer.adam.state_dict()['state'][0]['step'] == before['state'][0]['step']
    optimizer.step()
    torch.testing.assert_close(parameter, expected.model_parameter)
    assert other.master.item() == other.model_parameter.item() == 11.
