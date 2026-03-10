# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
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

# Pure FSDP topology (no EP): default 4 GPUs -> fsdp=2, dp=2.
fsdp_size = int(os.environ.get('FSDP_SIZE', '2'))
dp_size = int(os.environ.get('DP_SIZE', '2'))
device_mesh = DeviceMesh.from_sizes(fsdp_size=fsdp_size, dp_size=dp_size)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


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
        fsdp_config={'transformer_cls_names_to_wrap': ['Qwen3MoeSparseMoeBlock']},
    )

    # Full-parameter training: no LoRA adapter is added.
    model.set_optimizer(optimizer_cls='AdamW', lr=LR, foreach=False)
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
        f'dp_size={dp_size}, fsdp_size={fsdp_size}')
    if NUM_LAYERS is not None:
        logger.info(f'NUM_LAYERS={NUM_LAYERS}')

    for step, batch in enumerate(dataloader):
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
        model.clip_grad_and_step(
            max_grad_norm=MAX_GRAD_NORM,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        )
        if step % GRAD_ACCUM_STEPS == 0:
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(f'Current is step {step // GRAD_ACCUM_STEPS}, metric: {metric}')
        if step > 0 and step % 50 == 0:
            model.save('./output')


if __name__ == '__main__':
    train()
