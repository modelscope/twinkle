import pytest
import sys
import torch
import types
from torch import nn


def _ensure_dummy_zmq():
    if "zmq" in sys.modules:
        return
    sys.modules["zmq"] = types.SimpleNamespace(
        Context=object,
        Socket=object,
        RCVTIMEO=1,
        SNDTIMEO=2,
        LINGER=3,
    )


def test_ep_target_parameter_lora_gather_dim_matches_peft_flattening():
    _ensure_dummy_zmq()
    from twinkle.model.transformers.strategy.native_fsdp import _ep_expert_state_dict_gather_dim

    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts.lora_A.weight") == 0
    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts.base_layer.lora_A.weight") == 0
    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts.lora_B.weight") == 1
    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts.base_layer.lora_B.weight") == 1
    assert _ep_expert_state_dict_gather_dim(
        "model.layers.0.mlp.experts._twinkle_lora_gate_up_proj.lora_B.lora_0.weight") == 0


def test_ep_3d_expert_lora_gathers_both_factors_on_expert_dim():
    _ensure_dummy_zmq()
    from twinkle.model.transformers.strategy.native_fsdp import (
        _concat_ep_expert_shards,
        _ep_expert_state_dict_gather_dim,
    )

    name = "model.layers.0.mlp.experts.base_layer.lora_B.lora_0.weight"
    assert _ep_expert_state_dict_gather_dim(name, (8, 4096, 8), 8) == 0

    shards = [torch.full((8, 2, 1), rank) for rank in range(4)]
    full = _concat_ep_expert_shards(name, shards, {"experts_per_rank": 8, "num_experts": 32})
    assert full.shape == (32, 2, 1)
    assert torch.equal(full[:, 0, 0], torch.arange(4).repeat_interleave(8))


def test_ep_3d_expert_lora_load_splits_lora_b_on_expert_dim():
    _ensure_dummy_zmq()
    from twinkle.model.transformers.strategy.native_fsdp import _split_for_ep_pre_distribute

    class _Box(nn.Module):
        pass

    model = _Box()
    model.base_model = _Box()
    model.base_model.model = _Box()
    model.base_model.model.model = _Box()
    model.base_model.model.model.layers = nn.ModuleList([_Box()])
    mlp = _Box()
    model.base_model.model.model.layers[0].mlp = mlp
    mlp._ep_patched = True
    mlp.experts = _Box()
    mlp.experts.base_layer = _Box()
    mlp.experts.base_layer.lora_B = _Box()
    mlp.experts.base_layer.lora_B.lora_0 = _Box()

    key = 'base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.lora_0.weight'
    full = torch.arange(32).reshape(32, 1, 1).expand(32, 2, 1)
    local = _split_for_ep_pre_distribute(model, key, full, ep_world_size=4, ep_rank=2)

    assert local.shape == (8, 2, 1)
    assert torch.equal(local[:, 0, 0], torch.arange(16, 24))


class _FakeTensorExperts(nn.Module):

    def __init__(self, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(4, 3, 8, device=device, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(4, 4, 3, device=device, dtype=dtype))
        self.num_experts = 4


def test_target_parameter_lora_slots_stay_meta_until_fsdp_materialization():
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager

    model = nn.Module()
    model.experts = _FakeTensorExperts(device="meta")
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    manager.patch(model, ["experts.gate_up_proj", "experts.down_proj"])

    for wrapper in manager.wrappers:
        assert all(param.is_meta for param in wrapper.lora_A.values())
        assert all(param.is_meta for param in wrapper.lora_B.values())


def test_target_parameter_lora_defers_initial_snapshot_on_source_rank():
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager

    model = nn.Module()
    model.experts = _FakeTensorExperts()
    manager = TargetParameterLoraManager(max_loras=2, max_r=4, defer_initial_weights=True)
    manager.patch(model, ["experts.gate_up_proj", "experts.down_proj"])

    for wrapper in manager.wrappers:
        assert all(not param.is_meta for param in wrapper.lora_A.values())
        assert all(torch.count_nonzero(param) == 0 for param in wrapper.lora_B.values())
        assert wrapper._initial_lora_A == {}

    manager.save_initial_weights()

    for wrapper in manager.wrappers:
        assert set(wrapper._initial_lora_A) == {"lora_0", "lora_1"}


def test_target_parameter_lora_reuses_matching_preallocated_slots():
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager

    model = nn.Module()
    model.experts = _FakeTensorExperts()
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    targets = ["experts.gate_up_proj", "experts.down_proj"]

    manager.patch(model, targets)
    wrappers = list(manager.wrappers)
    manager.patch(model, targets)

    assert manager.patched_target_parameters == tuple(targets)
    assert manager.wrappers == wrappers

    with pytest.raises(ValueError, match="target_parameters already patched"):
        manager.patch(model, ["experts.gate_up_proj"])


def test_ep_shards_target_parameter_lora_slots_on_meta():
    _ensure_dummy_zmq()
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager
    from twinkle.model.transformers.moe.expert_parallel import _shard_tensor_experts

    model = nn.Module()
    model.experts = _FakeTensorExperts(device="meta")
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    manager.patch(model, ["experts.gate_up_proj", "experts.down_proj"])

    _shard_tensor_experts(model.experts, 2, 4)

    assert model.experts.gate_up_proj.shape[0] == 2
    assert model.experts.down_proj.shape[0] == 2
    for wrapper in manager.wrappers:
        assert wrapper.num_experts == 2
        assert all(param.shape[0] == 2 and param.is_meta for param in wrapper.lora_A.values())
        assert all(param.shape[0] == 2 and param.is_meta for param in wrapper.lora_B.values())


def test_ep_can_reuse_retained_full_tensor_storage_during_memory_efficient_init():
    _ensure_dummy_zmq()
    from twinkle.model.transformers.moe.expert_parallel import _shard_tensor_experts

    experts = _FakeTensorExperts()
    full_gate_up = experts.gate_up_proj
    full_down = experts.down_proj
    expected_gate_up = full_gate_up[2:4].detach().clone()
    expected_down = full_down[2:4].detach().clone()

    _shard_tensor_experts(experts, 2, 4, clone=False)

    assert experts.gate_up_proj.untyped_storage().data_ptr() == full_gate_up.untyped_storage().data_ptr()
    assert experts.down_proj.untyped_storage().data_ptr() == full_down.untyped_storage().data_ptr()
    assert torch.equal(experts.gate_up_proj, expected_gate_up)
    assert torch.equal(experts.down_proj, expected_down)


def test_ep_clones_expert_storage_by_default():
    _ensure_dummy_zmq()
    from twinkle.model.transformers.moe.expert_parallel import _shard_tensor_experts

    experts = _FakeTensorExperts()
    full_gate_up = experts.gate_up_proj
    full_down = experts.down_proj

    _shard_tensor_experts(experts, 2, 4)

    assert experts.gate_up_proj.untyped_storage().data_ptr() != full_gate_up.untyped_storage().data_ptr()
    assert experts.down_proj.untyped_storage().data_ptr() != full_down.untyped_storage().data_ptr()


def test_target_parameter_slot_reset_uses_materialized_ep_local_snapshot():
    _ensure_dummy_zmq()
    from peft import LoraConfig
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager
    from twinkle.model.transformers.moe.expert_parallel import _shard_tensor_experts

    torch.manual_seed(0)
    model = nn.Module()
    model.experts = _FakeTensorExperts()
    manager = TargetParameterLoraManager(max_loras=1, max_r=4)
    targets = ["experts.gate_up_proj", "experts.down_proj"]
    manager.patch(model, targets)
    manager.acquire(
        "tenant_a",
        "lora_0",
        LoraConfig(r=2, lora_alpha=4, target_modules=[], target_parameters=targets),
    )

    _shard_tensor_experts(model.experts, 2, 4)
    manager.save_initial_weights()
    initial_a = [wrapper.lora_A["lora_0"].detach().clone() for wrapper in manager.wrappers]

    with torch.no_grad():
        for wrapper in manager.wrappers:
            wrapper.lora_A["lora_0"].add_(1)
            wrapper.lora_B["lora_0"].add_(1)
    manager.release("tenant_a")

    for wrapper, expected_a in zip(manager.wrappers, initial_a):
        assert torch.equal(wrapper.lora_A["lora_0"], expected_a)
        assert torch.count_nonzero(wrapper.lora_B["lora_0"]) == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason="Need 4 GPUs")
def test_ep_fsdp_multi_lora_target_parameter_checkpoint_smoke():
    pytest.skip("Run this smoke in the DSV4 EP/FSDP integration environment with a local model fixture.")
