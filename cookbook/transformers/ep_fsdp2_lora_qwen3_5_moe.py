# Copyright (c) ModelScope Contributors. All rights reserved.
"""EP + FSDP2 + LoRA SFT cookbook for Qwen3.5-MoE.

Run on 8 GPUs:
    torchrun --nproc-per-node=8 cookbook/transformers/ep_fsdp2_lora_qwen3_5_moe.py
"""
import os
from pathlib import Path

from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.utils.framework import Torch
from twinkle.kernel import kernelize_model

logger = get_logger()

MODEL_ID = os.environ.get('QWEN3_MODEL_ID', 'ms://Qwen/Qwen3.6-35B-A3B')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'Qwen3_5Template')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '4'))
LOG_INTERVAL = GRAD_ACCUM_STEPS
LR = float(os.environ.get('LR', '1e-4'))
MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
LORA_R = int(os.environ.get('LORA_R', '8'))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', '32'))
ENABLE_EP = os.environ.get('ENABLE_EP', '1') == '1'
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')
RESUME_FROM_CHECKPOINT = os.environ.get('RESUME_FROM_CHECKPOINT') or None
RESUME_ONLY_MODEL = os.environ.get('RESUME_ONLY_MODEL', '0') == '1'
IGNORE_DATA_SKIP = os.environ.get('IGNORE_DATA_SKIP', '0') == '1'
ADAPTER_NAME = os.environ.get('ADAPTER_NAME', 'default')

device_mesh = DeviceMesh.from_sizes(
    fsdp_size=8,
    dp_size=1,
    ep_size=8,
    device_type=Platform.get_platform().device_prefix(),
)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def _build_lora_config(enable_ep: bool):
    if enable_ep:
        return LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules='all-linear',
            target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
        )
    # Expert weights are bare nn.Parameters. PEFT trains them through
    # target_parameters/ParamWrapper, which dynamically parametrizes weights
    # during forward. That is not stable with plain FSDP2, so non-EP mode uses
    # regular module LoRA and does not train expert parameters.
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules='all-linear',
    )


def save_checkpoint(model: TransformersModel, checkpoint_name: str, dataloader: DataLoader):
    return model.save(
        name=checkpoint_name,
        output_dir=OUTPUT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def train():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_config = getattr(config, 'text_config', config)
    if hasattr(text_config, 'use_cache'):
        text_config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID))
    try:
        dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    except ValueError:
        dataset.set_template('Qwen3_5Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle', 'ModelScope'))
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        fsdp_config={
            'expert_parallel': {
                'enabled': ENABLE_EP,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            }
        },
    )
    # npu patch
    if Torch.is_npu_available():
        model = kernelize_model(model, mode='train', device='npu')
    lora_cfg = _build_lora_config(ENABLE_EP)
    model.add_adapter_to_model(ADAPTER_NAME, lora_cfg, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    model.set_optimizer('AdamW', lr=LR, foreach=False)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
    )

    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = Path(RESUME_FROM_CHECKPOINT).expanduser().resolve()
        kwargs = {}
        if ADAPTER_NAME:
            kwargs['adapter_name'] = ADAPTER_NAME
        progress = model.resume_from_checkpoint(
            str(checkpoint_path), resume_only_model=RESUME_ONLY_MODEL, **kwargs)
        if not IGNORE_DATA_SKIP:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(
        f'Total steps: {len(dataloader)}, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}, '
        f'enable_ep={ENABLE_EP}, output_dir={OUTPUT_DIR}')

    optimizer_group = model.optimizer_group[ADAPTER_NAME]
    for batch in dataloader:
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step(max_grad_norm=MAX_GRAD_NORM, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
        cur_step = optimizer_group.cur_step
        if cur_step > 0 and cur_step % LOG_INTERVAL == 0:
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric}')

    final_checkpoint = save_checkpoint(model, 'checkpoint-final', dataloader)
    logger.info(f'Saved final adapter to {final_checkpoint}')


if __name__ == '__main__':
    train()
