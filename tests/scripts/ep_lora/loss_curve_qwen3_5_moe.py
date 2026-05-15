"""P2: EP+LoRA loss should decrease over 200 steps on self-cognition SFT.

Run on 4 GPUs:
    torchrun --nproc-per-node=4 tests/scripts/ep_lora/loss_curve_qwen3_5_moe.py
"""
import os

import torch.distributed as dist
from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
MODEL_ID = os.environ.get('QWEN3_MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TARGET_RATIO = float(os.environ.get('TARGET_RATIO', '0.7'))
NUM_STEPS = int(os.environ.get('NUM_STEPS', '200'))


def main():
    device_mesh = DeviceMesh.from_sizes(
        fsdp_size=4,
        dp_size=1,
        ep_size=2,
        device_type=Platform.get_platform().device_prefix(),
    )
    twinkle.initialize(mode='local', global_device_mesh=device_mesh)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = 4
    config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(1000)))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle', 'ModelScope'))
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=4, device_mesh=device_mesh)

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        fsdp_config={'expert_parallel': {'enabled': True, 'router_dtype': 'fp32'}},
    )
    model.add_adapter_to_model(
        'default',
        LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules='all-linear',
            target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
        ),
    )
    model.set_optimizer('AdamW', lr=1e-4, foreach=False)
    model.set_lr_scheduler('CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=NUM_STEPS)

    losses = []
    for step, batch in enumerate(dataloader):
        if step >= NUM_STEPS:
            break
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch, gradient_accumulation_steps=1)
        model.clip_grad_and_step(max_grad_norm=1.0, gradient_accumulation_steps=1)
        metric = model.calculate_metric(is_training=True)
        if callable(metric):
            metric = metric()
        loss = metric['loss'] if isinstance(metric, dict) and 'loss' in metric else metric
        losses.append(float(loss))

    if dist.get_rank() == 0:
        head = sum(losses[:10]) / 10
        tail = sum(losses[-10:]) / 10
        ratio = tail / head if head > 0 else float('inf')
        logger.info(f'head_avg={head:.4f}, tail_avg={tail:.4f}, ratio={ratio:.3f}')
        assert ratio < TARGET_RATIO, (
            f'loss did not decrease enough; tail/head ratio {ratio:.3f} >= {TARGET_RATIO}')
        logger.info('LOSS CURVE TEST PASSED')


if __name__ == '__main__':
    main()
