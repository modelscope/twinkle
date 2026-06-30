# Copyright (c) ModelScope Contributors. All rights reserved.
"""EP + FSDP2 + LoRA SFT cookbook for DeepSeek-V4.

Run on 8 GPUs:
    torchrun --nproc-per-node=8 cookbook/transformers/ep_fsdp2_lora_deepseek_v4.py
"""
from pathlib import Path

from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
args = CLI.from_args()

ENABLE_EP = args.extra.get('enable_ep', True)

device_mesh = DeviceMesh.from_sizes(
    fsdp_size=args.infra.fsdp_size,
    dp_size=args.infra.dp_size,
    ep_size=args.infra.ep_size,
    device_type=Platform.get_platform().device_prefix(),
)
twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)


def _build_lora_config():
    return LoraConfig(
        r=args.lora.lora_r,
        lora_alpha=args.lora.lora_alpha,
        exclude_modules=['o_a_proj'],
        target_modules='all-linear',
        target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
    )


def save_checkpoint(model: TransformersModel, checkpoint_name: str, dataloader: DataLoader):
    return model.save(
        name=checkpoint_name,
        output_dir=args.training.output_dir,
        adapter_name=args.lora.adapter_name,
        save_optimizer=args.checkpoint.save_optimizer,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def train():
    config = AutoConfig.from_pretrained(args.model.model_id, trust_remote_code=True)
    text_config = getattr(config, 'text_config', config)
    if hasattr(text_config, 'use_cache'):
        text_config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset.dataset_id))
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id)
    dataset.map(SelfCognitionProcessor(
        args.extra.get('model_name', 'twinkle'),
        args.extra.get('model_author', 'ModelScope'),
    ))
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size, device_mesh=device_mesh)

    model = TransformersModel(
        model_id=args.model.model_id,
        config=config,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        memory_efficient_init=args.model.memory_efficient_init,
        fsdp_config={
            'expert_parallel': {
                'enabled': ENABLE_EP,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            }
        },
    )
    lora_cfg = _build_lora_config()
    model.add_adapter_to_model(args.lora.adapter_name, lora_cfg,
                               gradient_accumulation_steps=args.training.gradient_accumulation_steps)
    model.set_optimizer(args.optimizer.optimizer_cls, lr=args.optimizer.learning_rate, foreach=False)
    model.set_lr_scheduler(
        scheduler_cls=args.scheduler.scheduler_cls,
        num_warmup_steps=args.scheduler.num_warmup_steps,
        num_training_steps=len(dataloader),
    )

    if args.training.resume_from_checkpoint:
        checkpoint_path = Path(args.training.resume_from_checkpoint).expanduser().resolve()
        kwargs = {}
        if args.lora.adapter_name:
            kwargs['adapter_name'] = args.lora.adapter_name
        progress = model.resume_from_checkpoint(
            str(checkpoint_path), resume_only_model=args.training.resume_only_model, **kwargs)
        if not args.training.ignore_data_skip:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(
        f'Total steps: {len(dataloader)}, batch_size={args.training.batch_size}, '
        f'grad_accum={args.training.gradient_accumulation_steps}, '
        f'enable_ep={ENABLE_EP}, output_dir={args.training.output_dir}')

    optimizer_group = model.optimizer_group[args.lora.adapter_name]
    for batch in dataloader:
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step(max_grad_norm=args.optimizer.max_grad_norm,
                                gradient_accumulation_steps=args.training.gradient_accumulation_steps)
        cur_step = optimizer_group.cur_step
        if cur_step > 0 and cur_step % args.training.log_interval == 0:
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric}')

    final_checkpoint = save_checkpoint(model, 'checkpoint-final', dataloader)
    logger.info(f'Saved final adapter to {final_checkpoint}')


if __name__ == '__main__':
    train()
