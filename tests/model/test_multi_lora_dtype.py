import sys
import torch
import types
from peft import LoraConfig
from torch import nn


def _ensure_dummy_zmq():
    if 'zmq' in sys.modules:
        return
    sys.modules['zmq'] = types.SimpleNamespace(
        Context=object,
        Socket=object,
        RCVTIMEO=1,
        SNDTIMEO=2,
        LINGER=3,
    )


def _make_multi_lora_model(device: str):
    _ensure_dummy_zmq()
    from twinkle.model.multi_lora import MultiLora

    model = nn.Sequential(nn.Linear(4, 4, device=device, dtype=torch.bfloat16))
    multi_lora = MultiLora(max_loras=2, max_r=4, defer_initial_weights=True)
    model = multi_lora.patch(
        model,
        target_modules=['0'],
        lora_config=LoraConfig(r=4, lora_alpha=8, target_modules=['0']),
    )
    return model


def _align_lora_dtype(model):
    _ensure_dummy_zmq()
    from twinkle.model.transformers.transformers import TransformersModel

    TransformersModel._ensure_lora_dtype(None, model)


def test_multi_lora_dtype_matches_bf16_base_before_fsdp_wrap():
    model = _make_multi_lora_model('cpu')

    assert {param.dtype for name, param in model.named_parameters() if 'lora_' in name} == {torch.float32}
    _align_lora_dtype(model)
    assert {param.dtype for name, param in model.named_parameters() if 'lora_' in name} == {torch.bfloat16}


def test_meta_multi_lora_dtype_matches_bf16_base_before_fsdp_wrap():
    model = _make_multi_lora_model('meta')

    assert all(param.is_meta for param in model.parameters())
    assert {param.dtype for name, param in model.named_parameters() if 'lora_' in name} == {torch.float32}
    _align_lora_dtype(model)
    assert {param.dtype for name, param in model.named_parameters() if 'lora_' in name} == {torch.bfloat16}
