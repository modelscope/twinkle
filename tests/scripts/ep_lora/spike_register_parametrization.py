"""P0 spike: verify PEFT ParamWrapper + FSDP2 DTensor compatibility on Qwen3.5-MoE.

Run on 4 GPUs:
    torchrun --nproc-per-node=4 tests/scripts/ep_lora/spike_register_parametrization.py
"""
import os

import torch
import torch.distributed as dist
from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_logger
from twinkle.model import TransformersModel

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
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 256
    config.moe_intermediate_size = 64
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.use_cache = False

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        fsdp_config={'expert_parallel': {'enabled': True, 'router_dtype': 'fp32'}},
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules='all-linear',
        target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
    )
    model.add_adapter_to_model('default', lora_cfg)
    model.set_optimizer('AdamW', lr=1e-4, foreach=False)

    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.manual_seed(42 + rank)
    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 16), device=Platform.get_local_device()),
        'labels': torch.randint(0, config.vocab_size, (2, 16), device=Platform.get_local_device()),
        'attention_mask': torch.ones(2, 16, dtype=torch.long, device=Platform.get_local_device()),
    }

    model.forward_backward(inputs=batch, gradient_accumulation_steps=1)
    metric = model.calculate_metric(is_training=True)
    if callable(metric):
        metric = metric()
    loss = metric['loss'] if isinstance(metric, dict) and 'loss' in metric else metric
    logger.info(f'spike loss (rank {rank}): {loss}')
    assert torch.is_tensor(loss) or isinstance(loss, float), 'loss should be a scalar'
    loss_val = float(loss)
    assert torch.isfinite(torch.tensor(loss_val)), f'loss not finite: {loss_val}'

    unwrapped = model.strategy.unwrap_model(model.model)
    lora_a_seen = 0
    lora_b_seen = 0
    base_grads = []
    for name, param in unwrapped.named_parameters():
        if 'experts' not in name:
            continue
        if 'lora_A' in name and param.grad is not None:
            lora_a_seen += 1
            assert param.grad.abs().sum().item() > 0, f'{name} grad is zero'
        if 'lora_B' in name and param.grad is not None:
            lora_b_seen += 1
        if 'base_layer.gate_up_proj' in name or 'base_layer.down_proj' in name:
            base_grads.append((name, param.grad))

    logger.info(f'lora_A grads seen: {lora_a_seen}, lora_B grads seen: {lora_b_seen}')
    assert lora_a_seen > 0, 'no lora_A grads observed under experts subtree'
    for name, grad in base_grads:
        assert grad is None, f'{name} should be frozen but has grad'

    if rank == 0:
        logger.info('SPIKE PASSED: PEFT ParamWrapper works with FSDP2 DTensor for routing experts.')


if __name__ == '__main__':
    main()
