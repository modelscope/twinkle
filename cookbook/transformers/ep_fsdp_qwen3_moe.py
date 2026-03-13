# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()

MODEL_ID = os.environ.get('QWEN3_MODEL_ID', 'ms://Qwen/Qwen3-30B-A3B-Instruct-2507')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'Template')
_num_layers_env = os.environ.get('NUM_LAYERS')
NUM_LAYERS = int(_num_layers_env) if _num_layers_env is not None else None
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '4'))
LR = float(os.environ.get('LR', '1e-5'))
MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
KEEP_ROUTER_LOGITS = os.environ.get('KEEP_ROUTER_LOGITS', '0') == '1'

# 8 gpus, dp=1, fsdp=8 (data parallel), ep_size=8 (expert parallel)
device_mesh = DeviceMesh.from_sizes(
    fsdp_size=8,
    dp_size=1,
    ep_size=8,
    device_type=Platform.get_platform().device_prefix(),
)

twinkle.initialize(
    mode='local',
    global_device_mesh=device_mesh,
)


def train():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if NUM_LAYERS is not None and hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = NUM_LAYERS
    if hasattr(config, 'use_cache'):
        config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(1000)))
    try:
        dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    except ValueError:
        dataset.set_template('Template', model_id=MODEL_ID)

    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode(batched=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        device_mesh=device_mesh,
    )

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        fsdp_config={
            'expert_parallel': {
                'enabled': True,
                'router_dtype': 'fp32',
                'keep_router_logits': KEEP_ROUTER_LOGITS,
            }
        },
    )
    # Disable foreach to avoid DTensor mixed-type errors in EP runs.
    model.set_optimizer('AdamW', lr=LR, foreach=False)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(
        f'Total steps: {len(dataloader)}, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}, '
        f'lr={LR:.2e}, max_grad_norm={MAX_GRAD_NORM}, '
        f'keep_router_logits={KEEP_ROUTER_LOGITS}')

    for step, batch in enumerate(dataloader):
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
        model.clip_grad_and_step(
            max_grad_norm=MAX_GRAD_NORM,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        )

        is_sync_step = ((step + 1) % GRAD_ACCUM_STEPS == 0)
        if is_sync_step:
            optimizer_step = (step + 1) // GRAD_ACCUM_STEPS
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(f'Current optimizer_step {optimizer_step}, metric: {metric}')
            if optimizer_step > 0 and optimizer_step % 50 == 0:
                model.save(name=f'checkpoint-step-{optimizer_step}', output_dir='./output')


if __name__ == '__main__':
    train()
