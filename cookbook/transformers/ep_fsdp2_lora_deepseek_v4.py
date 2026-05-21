# Copyright (c) ModelScope Contributors. All rights reserved.
"""EP + FSDP2 + LoRA SFT cookbook for DeepSeek-V4.

Run on 4 GPUs:
    torchrun --nproc-per-node=4 cookbook/transformers/ep_fsdp2_lora_deepseek_v4.py
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

logger = get_logger()

MODEL_ID = os.environ.get('DSV4_MODEL_ID', 'ms://deepseek-ai/DeepSeek-V4')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'DeepseekV4Template')
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', '2'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '4'))
LR = float(os.environ.get('LR', '1e-4'))
MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
LORA_R = int(os.environ.get('LORA_R', '8'))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', '32'))
ENABLE_EP = os.environ.get('ENABLE_EP', '1') == '1'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', '0'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output_dsv4')
RESUME_FROM_CHECKPOINT = os.environ.get('RESUME_FROM_CHECKPOINT') or None
RESUME_ONLY_MODEL = os.environ.get('RESUME_ONLY_MODEL', '0') == '1'
IGNORE_DATA_SKIP = os.environ.get('IGNORE_DATA_SKIP', '0') == '1'
ADAPTER_NAME = os.environ.get('ADAPTER_NAME', 'default')

device_mesh = DeviceMesh.from_sizes(
    fsdp_size=4,
    dp_size=1,
    ep_size=2,
    device_type=Platform.get_platform().device_prefix(),
)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def _get_text_config(config):
    return getattr(config, 'text_config', config)


def _configure_smoke_config(config):
    text_config = _get_text_config(config)
    old_num_hidden_layers = getattr(text_config, 'num_hidden_layers', NUM_LAYERS)
    text_config.num_hidden_layers = NUM_LAYERS
    if hasattr(text_config, 'use_cache'):
        text_config.use_cache = False
    if hasattr(text_config, 'num_hash_layers'):
        text_config.num_hash_layers = min(text_config.num_hash_layers, NUM_LAYERS)
    if hasattr(text_config, 'compress_ratios'):
        extra_entries = max(len(text_config.compress_ratios) - old_num_hidden_layers, 0)
        keep = min(len(text_config.compress_ratios), NUM_LAYERS + extra_entries)
        text_config.compress_ratios = list(text_config.compress_ratios[:keep])


def _build_lora_config(enable_ep: bool):
    if enable_ep:
        return LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules='all-linear',
            exclude_modules=['o_a_proj'],
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
    model.save(
        name=checkpoint_name,
        output_dir=OUTPUT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def train():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    _configure_smoke_config(config)

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(500)))
    dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle', 'ModelScope'))
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        memory_efficient_init=True,
        fsdp_config={
            'expert_parallel': {
                'enabled': ENABLE_EP,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            }
        },
    )
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
        f'num_layers={NUM_LAYERS}, enable_ep={ENABLE_EP}, save_steps={SAVE_STEPS}, output_dir={OUTPUT_DIR}')

    optimizer_group = model.optimizer_group[ADAPTER_NAME]
    for step, batch in enumerate(dataloader):
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step(max_grad_norm=MAX_GRAD_NORM, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
        cur_step = optimizer_group.cur_step
        if cur_step > 0 and (step + 1) % GRAD_ACCUM_STEPS == 0:
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(f'optimizer_step {cur_step}, metric: {metric}')
            if SAVE_STEPS and cur_step % SAVE_STEPS == 0:
                save_checkpoint(model, f'checkpoint-{cur_step}', dataloader)

    save_checkpoint(model, 'checkpoint-final', dataloader)
    logger.info(f'Saved final adapter to {OUTPUT_DIR}/checkpoint-final')


if __name__ == '__main__':
    train()
