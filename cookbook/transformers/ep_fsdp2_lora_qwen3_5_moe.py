# Copyright (c) ModelScope Contributors. All rights reserved.
"""EP + FSDP2 + LoRA SFT cookbook for Qwen3.5-MoE.

Run on 8 GPUs:
    torchrun --nproc-per-node=8 cookbook/transformers/ep_fsdp2_lora_qwen3_5_moe.py
"""
from pathlib import Path
import os

from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.utils.framework import Torch
from twinkle.kernel import kernelize, liger_builtin, npu_builtin

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

    # Slice the dataset to the training budget (default 320, matching the other
    # cookbooks) so encoding the full LongAlpaca set (~12k examples) doesn't
    # dominate runtime; this is a data-amount knob, not a sharding/batch change.
    _train_samples = args.training.train_samples or 320
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=range(_train_samples)))
    try:
        dataset.set_template(args.template.template_cls, model_id=args.model.model_id,
                             max_length=args.template.max_length,
                             truncation_strategy=args.template.truncation_strategy)
    except ValueError:
        dataset.set_template('Qwen3_5Template', model_id=args.model.model_id,
                            max_length=args.template.max_length,
                            truncation_strategy=args.template.truncation_strategy)
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
        fsdp_config={
            'expert_parallel': {
                'enabled': ENABLE_EP,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            }
        },
    )
    # Kernel mode: torch (TWINKLE_TORCH_BASELINE=1, no fusion) | npu (default,
    # CANN + FLA) | npu+liger (--enable-liger, Liger per-layer on top of CANN).
    # Sharding/batch are unchanged across modes.
    _torch_baseline = os.environ.get('TWINKLE_TORCH_BASELINE', '').lower() in ('1', 'true', 'yes')
    kernel_mapping = {}
    if Torch.is_npu_available() and not _torch_baseline:
        kernel_mapping.update(npu_builtin(model))
    if args.model.enable_liger and not _torch_baseline:
        kernel_mapping.update(liger_builtin(model))
    if kernel_mapping:
        model = kernelize(model, kernel_mapping)
    _use_fused_ce = args.model.enable_liger and not _torch_baseline and args.model.enable_fused_ce
    _task = 'fused_lm_ce' if _use_fused_ce else 'causal_lm'
    lora_cfg = _build_lora_config()
    model.add_adapter_to_model(args.lora.adapter_name, lora_cfg,
                               gradient_accumulation_steps=args.training.gradient_accumulation_steps)
    model.set_optimizer(args.optimizer.optimizer_cls, lr=args.optimizer.learning_rate, foreach=False)
    model.set_lr_scheduler(
        scheduler_cls=args.scheduler.scheduler_cls,
        num_warmup_steps=args.scheduler.num_warmup_steps,
        num_training_steps=len(dataloader),
    )
    if _use_fused_ce:
        model.set_loss('LigerFusedLinearCrossEntropyLoss', adapter_name=args.lora.adapter_name,
                       reduction='sum')

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
    if _use_fused_ce:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.barrier()
        if Torch.is_npu_available():
            import torch_npu
            torch.npu.synchronize()
    for batch in dataloader:
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch, task=_task)
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
