"""Verify _patch_adapter triggers EP slicing before get_peft_model.

Run on 4 GPUs:
    torchrun --nproc-per-node=4 tests/scripts/ep_lora/test_patch_adapter_ep_ordering.py
"""
import os

from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_logger
from twinkle.model import TransformersModel

from ep_lora_config_helpers import set_text_config_attrs

logger = get_logger()
MODEL_ID = os.environ.get('QWEN3_MODEL_ID', 'ms://Qwen/Qwen3.5-4B')


def main():
    device_mesh = DeviceMesh.from_sizes(
        fsdp_size=4,
        dp_size=1,
        ep_size=2,
        device_type=Platform.get_platform().device_prefix(),
    )
    twinkle.initialize(mode='local', global_device_mesh=device_mesh)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    set_text_config_attrs(config, num_hidden_layers=2, num_experts=4, use_cache=False)

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        fsdp_config={'expert_parallel': {'enabled': True}},
    )
    assert not getattr(model, '_expert_parallel_applied', False)

    model.add_adapter_to_model(
        'default',
        LoraConfig(
            r=4,
            target_modules='all-linear',
            target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
        ),
    )
    assert getattr(model, '_expert_parallel_applied', False), (
        '_patch_adapter must trigger EP slicing before PEFT wrap')

    unwrapped = model.strategy.unwrap_model(model.model)
    lora_a_shapes = [(n, p.shape) for n, p in unwrapped.named_parameters() if 'experts' in n and 'lora_A' in n]
    assert lora_a_shapes, 'no lora_A weight under experts subtree'
    for name, shape in lora_a_shapes:
        assert shape[0] == 4 * 2, f'{name} shape {tuple(shape)}; expected r*E_local=8'

    logger.info('ORDERING TEST PASSED')


if __name__ == '__main__':
    main()
