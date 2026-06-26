# Copyright (c) ModelScope Contributors. All rights reserved.
"""EP + FSDP2 + Multi-LoRA SFT cookbook for DeepSeek-V4.

Run on 8 GPUs:
    torchrun --nproc-per-node=8 cookbook/transformers/ep_fsdp2_multi_lora_deepseek_v4.py
"""
import os
from pathlib import Path

from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MultiLoraTransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()

MODEL_ID = os.environ.get('DSV4_MODEL_ID', 'ms://deepseek-ai/DeepSeek-V4-Flash')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'DeepseekV4Template')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '4'))
LOG_INTERVAL = GRAD_ACCUM_STEPS
LR = float(os.environ.get('LR', '1e-4'))
MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
LORA_R = int(os.environ.get('LORA_R', '8'))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', '32'))
MAX_LORAS = int(os.environ.get('MAX_LORAS', '2'))
MAX_R = int(os.environ.get('MAX_R', str(max(32, LORA_R))))
ENABLE_EP = os.environ.get('ENABLE_EP', '1') == '1'
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output_dsv4_multi_lora')
RESUME_FROM_CHECKPOINT = os.environ.get('RESUME_FROM_CHECKPOINT') or None
RESUME_ONLY_MODEL = os.environ.get('RESUME_ONLY_MODEL', '0') == '1'
IGNORE_DATA_SKIP = os.environ.get('IGNORE_DATA_SKIP', '0') == '1'
ADAPTER_NAMES = [name.strip() for name in os.environ.get('ADAPTER_NAMES', 'tenant_a,tenant_b').split(',') if name]

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
            exclude_modules=['o_a_proj'],
            target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
        )
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        exclude_modules=['o_a_proj'],
        target_modules='all-linear',
    )


def save_checkpoint(model: MultiLoraTransformersModel, adapter_name: str, dataloader: DataLoader):
    return model.save(
        name=f'checkpoint-final-{adapter_name}',
        output_dir=OUTPUT_DIR,
        adapter_name=adapter_name,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def train():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_config = getattr(config, 'text_config', config)
    if hasattr(text_config, 'use_cache'):
        text_config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID))
    dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle', 'ModelScope'))
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)

    ep_lora_cfg = _build_lora_config(enable_ep=ENABLE_EP) # LoraConfig for target params
    lora_cfg = _build_lora_config(enable_ep=False)  # LoraConfig for PEFT adapter
    model = MultiLoraTransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        memory_efficient_init=True,
        max_loras=MAX_LORAS,
        max_r=MAX_R,
        fsdp_config={
            'expert_parallel': {
                'enabled': ENABLE_EP,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            }
        },
        lora_config=lora_cfg,
    )
    
    for adapter_name in ADAPTER_NAMES:
        model.add_adapter_to_model(adapter_name, ep_lora_cfg, gradient_accumulation_steps=GRAD_ACCUM_STEPS)

    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = Path(RESUME_FROM_CHECKPOINT).expanduser().resolve()
        for adapter_name in ADAPTER_NAMES:
            progress = model.resume_from_checkpoint(
                str(checkpoint_path),
                resume_only_model=RESUME_ONLY_MODEL,
                adapter_name=adapter_name,
            )
        if not IGNORE_DATA_SKIP:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])

    logger.info(get_device_placement())
    for adapter_name in ADAPTER_NAMES:
        logger.info(model.get_train_configs(adapter_name=adapter_name))
    logger.info(
        f'Total steps: {len(dataloader)}, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}, '
        f'enable_ep={ENABLE_EP}, adapters={ADAPTER_NAMES}, output_dir={OUTPUT_DIR}')

    # After LoRA init, before forward (LoRA active): perform EP + FSDP broadcast & sharding.
    model._lazy_wrap_model()

    # Must call set_optimizer() after EP + FSDP sharding, otherwise optimizer may
    # capture stale parameter references and fail to update the actual LoRA weights.
    for adapter_name in ADAPTER_NAMES:
        model.set_optimizer('AdamW', lr=LR, foreach=False, adapter_name=adapter_name)
        model.set_lr_scheduler(
            scheduler_cls='CosineWarmupScheduler',
            num_warmup_steps=5,
            num_training_steps=len(dataloader),
            adapter_name=adapter_name,
        )
    
    for batch_idx, batch in enumerate(dataloader):
        if callable(batch):
            batch = batch()
        adapter_name = ADAPTER_NAMES[batch_idx % len(ADAPTER_NAMES)]
        model.forward_backward(
            inputs=batch,
            adapter_name=adapter_name,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        )
        model.clip_grad_and_step(
            max_grad_norm=MAX_GRAD_NORM,
            adapter_name=adapter_name,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        )
        cur_step = model.optimizer_group[adapter_name].cur_step
        if cur_step > 0 and cur_step % LOG_INTERVAL == 0:
            metric = model.calculate_metric(is_training=True, adapter_name=adapter_name)
            if callable(metric):
                metric = metric()
            logger.info(f'Adapter {adapter_name} is at step {cur_step} of {len(dataloader)}, metric: {metric}')

    for adapter_name in ADAPTER_NAMES:
        checkpoint = save_checkpoint(model, adapter_name, dataloader)
        logger.info(f'Saved final adapter {adapter_name} to {checkpoint}')


if __name__ == '__main__':
    train()
