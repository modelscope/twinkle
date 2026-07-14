import pytest
import sys
import torch
import types


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
    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts._twinkle_lora_gate_up_proj.lora_B.lora_0.weight") == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason="Need 4 GPUs")
def test_ep_fsdp_multi_lora_target_parameter_checkpoint_smoke():
    pytest.skip("Run this smoke in the DSV4 EP/FSDP integration environment with a local model fixture.")
