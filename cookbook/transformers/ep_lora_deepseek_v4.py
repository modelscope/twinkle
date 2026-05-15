# Copyright (c) ModelScope Contributors. All rights reserved.
"""EP + LoRA SFT cookbook for DeepSeek-V4.

Run on 4 GPUs:
    torchrun --nproc-per-node=4 cookbook/transformers/ep_lora_deepseek_v4.py
"""
import os

from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()

MODEL_ID = os.environ.get('DSV4_MODEL_ID', 'ms://deepseek-ai/DeepSeek-V4')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'DeepseekV4Template')
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', '2'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '2'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '4'))
LR = float(os.environ.get('LR', '1e-4'))
MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
LORA_R = int(os.environ.get('LORA_R', '8'))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', '32'))
NUM_STEPS_LIMIT = int(os.environ.get('NUM_STEPS_LIMIT', '0'))

device_mesh = DeviceMesh.from_sizes(
    fsdp_size=4,
    dp_size=1,
    ep_size=2,
    device_type=Platform.get_platform().device_prefix(),
)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def train():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = NUM_LAYERS
    if hasattr(config, 'use_cache'):
        config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(500)))
    dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle', 'ModelScope'))
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        fsdp_config={'expert_parallel': {'enabled': True, 'router_dtype': 'fp32'}},
    )
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules='all-linear',
        target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
    )
    model.add_adapter_to_model('default', lora_cfg)
    model.set_optimizer('AdamW', lr=LR, foreach=False)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())

    for step, batch in enumerate(dataloader):
        if NUM_STEPS_LIMIT and step >= NUM_STEPS_LIMIT:
            break
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
        model.clip_grad_and_step(max_grad_norm=MAX_GRAD_NORM, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer_step = (step + 1) // GRAD_ACCUM_STEPS
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(f'optimizer_step {optimizer_step}, metric: {metric}')

    model.save(name='checkpoint-final', output_dir='./output_dsv4')


if __name__ == '__main__':
    train()
