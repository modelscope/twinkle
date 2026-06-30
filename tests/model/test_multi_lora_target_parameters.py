import copy
import sys
import types

import torch
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from torch import nn


class FakePackedExperts(nn.Module):

    def __init__(self, num_experts=2, hidden=4, intermediate=6, *, is_transposed=False):
        super().__init__()
        self.is_transposed = is_transposed
        if is_transposed:
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, intermediate * 2, hidden))
            self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
        else:
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate * 2))
            self.down_proj = nn.Parameter(torch.randn(num_experts, intermediate, hidden))

    def forward(self, x, expert_idx=0):
        gate_up = self.gate_up_proj[expert_idx]
        down = self.down_proj[expert_idx]
        if self.is_transposed:
            hidden = torch.nn.functional.linear(x, gate_up)
            gate, up = hidden.chunk(2, dim=-1)
            return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down)

        hidden = torch.nn.functional.linear(x, gate_up.T)
        gate, up = hidden.chunk(2, dim=-1)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down.T)


class FakeModel(nn.Module):

    def __init__(self, *, is_transposed=False):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = FakePackedExperts(is_transposed=is_transposed)

    def forward(self, x, expert_idx=0):
        return self.mlp.experts(x, expert_idx=expert_idx)


def test_peft_target_parameter_key_shapes_for_3d_experts():
    model = FakeModel()
    cfg = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=[],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )

    peft_model = get_peft_model(model, cfg, adapter_name="default")
    state = peft_model.state_dict()
    lora_shapes = {key: tuple(state[key].shape) for key in state if "lora_" in key}

    assert lora_shapes == {
        "base_model.model.mlp.experts.base_layer.lora_A.default.weight": (4, 12),
        "base_model.model.mlp.experts.base_layer.lora_B.default.weight": (4, 4),
        "base_model.model.mlp.experts.lora_A.default.weight": (4, 4),
        "base_model.model.mlp.experts.lora_B.default.weight": (6, 4),
    }


def _make_target_cfg(r=2):
    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=[],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )


def test_target_parameter_multi_lora_updates_only_active_adapter():
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager

    torch.manual_seed(0)
    model = FakeModel()
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    manager.patch(model, target_parameters=_make_target_cfg().target_parameters)
    manager.acquire("adapter_a", "lora_0", _make_target_cfg(r=2))
    manager.acquire("adapter_b", "lora_1", _make_target_cfg(r=2))

    params_before = {
        name: param.detach().clone()
        for name, param in manager.named_slot_parameters("adapter_b")
    }

    opt = torch.optim.SGD(manager.parameters_for_tenant("adapter_a"), lr=0.1)
    with manager.adapter("adapter_a"):
        loss = model(torch.randn(3, 4), expert_idx=0).pow(2).mean()
    loss.backward()
    opt.step()

    for name, param in manager.named_slot_parameters("adapter_b"):
        assert torch.equal(param.detach(), params_before[name])


def test_multilora_releases_target_parameter_slot_to_initial_weights():
    from twinkle.model.multi_lora import LoraTenant, MultiLora

    torch.manual_seed(0)
    model = FakeModel()
    multi_lora = MultiLora(max_loras=2, max_r=4)
    multi_lora.module = model
    multi_lora.loras = [
        LoraTenant(index=0, adapter_name="lora_0", config=_make_target_cfg(r=4)),
        LoraTenant(index=1, adapter_name="lora_1", config=_make_target_cfg(r=4)),
    ]
    multi_lora.patch_target_parameters(model, _make_target_cfg().target_parameters)
    multi_lora.acquire_lora("adapter_a", _make_target_cfg(r=2))

    initial_a = {
        name: param.detach().clone()
        for name, param in multi_lora.target_parameter_manager.named_slot_parameters("adapter_a")
        if ".lora_A." in name
    }

    with torch.no_grad():
        for _, param in multi_lora.target_parameter_manager.named_slot_parameters("adapter_a"):
            param.add_(1.0)

    multi_lora.release_lora("adapter_a")

    for wrapper in multi_lora.target_parameter_manager.wrappers:
        for name, param in wrapper.named_slot_parameters("lora_0"):
            if ".lora_A." in name:
                assert torch.equal(param.detach(), initial_a[name])
            else:
                assert torch.count_nonzero(param.detach()) == 0


def test_target_parameter_state_dict_loads_with_peft():
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager

    torch.manual_seed(0)
    model = FakeModel()
    base_state = copy.deepcopy(model.state_dict())
    cfg = _make_target_cfg(r=2)
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    manager.patch(model, target_parameters=cfg.target_parameters)
    manager.acquire("adapter_a", "lora_0", cfg)

    with torch.no_grad():
        for _, param in manager.named_slot_parameters("adapter_a"):
            param.uniform_(-0.1, 0.1)

    inputs = torch.randn(3, 4)
    with manager.adapter("adapter_a"):
        expected = model(inputs, expert_idx=0)
    state = manager.get_state_dict("adapter_a")

    peft_model = FakeModel()
    peft_model.load_state_dict(base_state)
    peft_model = get_peft_model(peft_model, cfg, adapter_name="default")
    set_peft_model_state_dict(peft_model, state, adapter_name="default")

    actual = peft_model(inputs, expert_idx=0)

    assert torch.allclose(actual, expected, atol=1e-6)


def _make_multilora_for_target_parameters(model):
    from twinkle.model.multi_lora import LoraTenant, MultiLora

    model.active_adapter = "lora_0"
    model.set_adapter = lambda adapter_name: setattr(model, "active_adapter", adapter_name)
    model.disable_adapter_layers = lambda: None
    model.enable_adapter_layers = lambda: None
    multi_lora = MultiLora(max_loras=2, max_r=4)
    multi_lora.module = model
    multi_lora.loras = [
        LoraTenant(index=0, adapter_name="lora_0", config=_make_target_cfg(r=4)),
        LoraTenant(index=1, adapter_name="lora_1", config=_make_target_cfg(r=4)),
    ]
    multi_lora.patch_target_parameters(model, _make_target_cfg().target_parameters)
    return multi_lora


def test_multilora_state_dict_round_trips_target_parameters():
    torch.manual_seed(0)
    model = FakeModel()
    base_state = copy.deepcopy(model.state_dict())
    multi_lora = _make_multilora_for_target_parameters(model)
    multi_lora.acquire_lora("adapter_a", _make_target_cfg(r=2))

    with torch.no_grad():
        for _, param in multi_lora.target_parameter_manager.named_slot_parameters("adapter_a"):
            param.uniform_(-0.1, 0.1)

    inputs = torch.randn(3, 4)
    with multi_lora.adapter("adapter_a"):
        expected = model(inputs, expert_idx=0)
    state = multi_lora.get_state_dict("adapter_a")

    restored_model = FakeModel()
    restored_model.load_state_dict(base_state)
    restored = _make_multilora_for_target_parameters(restored_model)
    restored.acquire_lora("adapter_a", _make_target_cfg(r=2))
    restored.set_state_dict("adapter_a", state)

    with restored.adapter("adapter_a"):
        actual = restored_model(inputs, expert_idx=0)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_multilora_transformers_installs_target_parameters_once():
    from twinkle.model.multi_lora import LoraTenant, MultiLora

    if "zmq" not in sys.modules:
        sys.modules["zmq"] = types.SimpleNamespace(
            Context=object,
            Socket=object,
            RCVTIMEO=1,
            SNDTIMEO=2,
            LINGER=3,
        )
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel

    model = FakeModel()
    instance = MultiLoraTransformersModel.__new__(MultiLoraTransformersModel)
    nn.Module.__init__(instance)
    instance.model = model
    instance._model_wrapped = False
    instance._enable_expert_parallel = False
    instance.multi_adapter = MultiLora(max_loras=2, max_r=4)
    instance.multi_adapter.module = model
    instance.multi_adapter.loras = [
        LoraTenant(index=0, adapter_name="lora_0", config=_make_target_cfg(r=4)),
        LoraTenant(index=1, adapter_name="lora_1", config=_make_target_cfg(r=4)),
    ]

    cfg = _make_target_cfg(r=2)
    instance._ensure_target_parameter_lora_installed(cfg)
    instance.multi_adapter.acquire_lora("adapter_a", cfg)
    instance._ensure_target_parameter_lora_installed(cfg)
    instance.multi_adapter.acquire_lora("adapter_b", cfg)

    assert len(instance.multi_adapter.target_parameter_manager.wrappers) == 2

    different_cfg = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=[],
        target_parameters=["mlp.experts.gate_up_proj"],
    )
    try:
        instance._ensure_target_parameter_lora_installed(different_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("different target_parameters should be rejected")
