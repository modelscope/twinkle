import torch
from peft import LoraConfig
from torch import nn

from twinkle.model.multi_lora import MultiLora
from twinkle.model.transformers.transformers import TransformersModel


def test_multi_lora_dtype_matches_bf16_base_before_fsdp_wrap():
    model = nn.Sequential(nn.Linear(4, 4, dtype=torch.bfloat16))
    multi_lora = MultiLora(max_loras=2, max_r=4)
    model = multi_lora.patch(
        model,
        target_modules=['0'],
        lora_config=LoraConfig(r=4, lora_alpha=8, target_modules=['0']),
    )

    assert {param.dtype for name, param in model.named_parameters() if 'lora_' in name} == {torch.float32}

    TransformersModel._ensure_lora_dtype(None, model)

    assert {param.dtype for name, param in model.named_parameters() if 'lora_' in name} == {torch.bfloat16}
