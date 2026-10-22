import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
from twinkle.model.transformers.transformers import TransformersModel


@pytest.mark.parametrize('weights_only', [True, False])
def test_full_resume_restores_weights_and_optional_training_state(tmp_path, weights_only):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    source = torch.nn.Parameter(torch.ones(1, 1))
    saved_optimizer = torch.optim.Adam([source], lr=0.1)
    source.grad = torch.full_like(source, 5)
    saved_optimizer.step()
    torch.save(saved_optimizer.state_dict(), tmp_path / 'optimizer.pt')
    save_file({'weight': torch.full((1, 1), 2.)}, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'trainer_state.json').write_text(json.dumps({
        'cur_step': 3, 'consumed_train_samples': 12, 'gradient_accumulation_steps': 2,
    }))
    wrapper = object.__new__(TransformersModel)
    wrapper.__dict__['model'] = model
    wrapper.strategy = NativeFSDPStrategy(device_mesh=None, enable_ep=False)
    wrapper.optimizer_group = {'': SimpleNamespace(optimizer=optimizer, lr_scheduler=None, scaler=None,
                                                   cur_step=0, gradient_accumulation_steps=1)}
    result = wrapper.resume_from_checkpoint(str(tmp_path), resume_only_model=weights_only)
    assert model(torch.ones(1, 1)).item() == 2
    assert result['cur_step'] == 3
    if weights_only:
        assert not optimizer.state
        assert wrapper.optimizer_group[''].cur_step == 0
    else:
        torch.testing.assert_close(optimizer.state[model.weight]['exp_avg'], torch.full((1, 1), 0.5))
        assert wrapper.optimizer_group[''].cur_step == 3
        assert wrapper.optimizer_group[''].gradient_accumulation_steps == 2
