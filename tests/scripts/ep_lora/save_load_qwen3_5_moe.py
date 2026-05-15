"""P1: EP+LoRA save-reload numeric consistency for Qwen3.5-MoE.

Run on 4 GPUs:
    torchrun --nproc-per-node=4 tests/scripts/ep_lora/save_load_qwen3_5_moe.py
"""
import os
import shutil
import tempfile

import torch
import torch.distributed as dist
from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_logger
from twinkle.model import TransformersModel

logger = get_logger()
MODEL_ID = os.environ.get('QWEN3_MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
TOL = float(os.environ.get('TOL', '1e-3'))


def build_model():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = 2
    config.use_cache = False
    device_mesh = DeviceMesh.current()
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
    return model, config


def fixed_batch(config):
    torch.manual_seed(0)
    return {
        'input_ids': torch.randint(0, config.vocab_size, (2, 32), device=Platform.get_local_device()),
        'labels': torch.randint(0, config.vocab_size, (2, 32), device=Platform.get_local_device()),
        'attention_mask': torch.ones(2, 32, dtype=torch.long, device=Platform.get_local_device()),
    }


def compute_loss(model, batch):
    model.forward_backward(inputs=batch, gradient_accumulation_steps=1)
    metric = model.calculate_metric(is_training=True)
    if callable(metric):
        metric = metric()
    loss = metric['loss'] if isinstance(metric, dict) and 'loss' in metric else metric
    return float(loss)


def main():
    device_mesh = DeviceMesh.from_sizes(
        fsdp_size=4,
        dp_size=1,
        ep_size=2,
        device_type=Platform.get_platform().device_prefix(),
    )
    twinkle.initialize(mode='local', global_device_mesh=device_mesh)

    model_a, config = build_model()
    batch = fixed_batch(config)
    model_a.forward_backward(inputs=batch, gradient_accumulation_steps=1)
    model_a.clip_grad_and_step(max_grad_norm=1.0, gradient_accumulation_steps=1)
    loss_before = compute_loss(model_a, batch)

    tmp_root = tempfile.mkdtemp(prefix='ep_lora_save_')
    if dist.get_rank() == 0:
        logger.info(f'Save root: {tmp_root}')
    model_a.save(name='ckpt', output_dir=tmp_root)
    dist.barrier()

    del model_a
    torch.cuda.empty_cache()

    model_b, _ = build_model()
    model_b.load(name='ckpt', output_dir=tmp_root)
    loss_after = compute_loss(model_b, batch)

    diff = abs(loss_after - loss_before)
    if dist.get_rank() == 0:
        logger.info(f'loss_before={loss_before:.6f}, loss_after={loss_after:.6f}, diff={diff:.6e}')
    assert diff < TOL, f'save/reload loss drift {diff:.3e} > tol {TOL:.1e}'

    if dist.get_rank() == 0:
        shutil.rmtree(tmp_root, ignore_errors=True)
        logger.info('SAVE/LOAD TEST PASSED')


if __name__ == '__main__':
    main()
