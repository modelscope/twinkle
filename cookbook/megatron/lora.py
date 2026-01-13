# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-Core LoRA training example.

Usage (8 GPUs with TP2 PP2 CP2):
    torchrun --nproc_per_node=8 cookbook/megatron/lora.py --tp_size 2 --pp_size 2 --cp_size 2

Usage (4 GPUs with TP2 PP2):
    torchrun --nproc_per_node=4 cookbook/megatron/lora.py --tp_size 2 --pp_size 2

Usage (single GPU):
    torchrun --nproc_per_node=1 cookbook/megatron/lora.py --tp_size 1 --pp_size 1
"""
import argparse
import os

# CRITICAL: Set CUDA device before any CUDA imports to ensure correct device placement
import torch
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))
torch.cuda.set_device(LOCAL_RANK)

import numpy as np
from peft import LoraConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

import twinkle
from twinkle import get_device_placement, get_logger, DeviceMesh, DeviceGroup, Platform
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import VocabParallelCrossEntropyLoss
from twinkle.model import MegatronModel
from twinkle.processor import InputProcessor

logger = get_logger()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tp_size', type=int, default=1)
parser.add_argument('--pp_size', type=int, default=1)
parser.add_argument('--cp_size', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=None)
args = parser.parse_args()

# Get parallelism config
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
TP_SIZE = args.tp_size
PP_SIZE = args.pp_size
CP_SIZE = args.cp_size
DP_SIZE = WORLD_SIZE // (TP_SIZE * PP_SIZE * CP_SIZE)

# Device mesh: Match Megatron's order "tp-cp-ep-dp-pp" from innermost to outermost
# For mesh shape, we reverse the order: (pp, dp, cp, tp) where rightmost is innermost
# This ensures DP groups match between twinkle and Megatron
device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.arange(WORLD_SIZE).reshape(PP_SIZE, DP_SIZE, CP_SIZE, TP_SIZE),
    mesh_dim_names=('pp', 'dp', 'cp', 'tp'),
)

device_group = [
    DeviceGroup(
        name='model',
        ranks=list(range(WORLD_SIZE)),
        device_type=Platform.get_platform().device_prefix(),
    )
]

twinkle.initialize(
    mode='local',
    nproc_per_node=WORLD_SIZE,
    groups=device_group,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)


def create_dataset():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True, load_from_cache_file=False)
    return dataset


def train():
    # Use smaller batch size for single GPU to avoid OOM
    batch_size = 2 if WORLD_SIZE == 1 else 8
    dataloader = DataLoader(dataset=create_dataset, batch_size=batch_size)

    model = MegatronModel(
        pretrained_model_name_or_path='ms://Qwen/Qwen2.5-7B-Instruct',
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=PP_SIZE,
        context_parallel_size=CP_SIZE,
        mixed_precision='bf16',
        # Use 'full' recompute for single GPU to reduce memory usage
        recompute_granularity='full' if WORLD_SIZE <= 2 else 'selective',
    )

    lora_config = LoraConfig(target_modules='all-linear')

    # Use 'lora' as adapter_name and pass it consistently to all methods
    adapter_name = 'lora'
    model.add_adapter_to_model(adapter_name, lora_config, gradient_accumulation_steps=16)
    model.set_template('Qwen3Template', adapter_name=adapter_name)
    model.set_processor(InputProcessor, padding_side='right', adapter_name=adapter_name)
    # Note: For MegatronModel, loss is computed internally by Megatron.
    # set_loss() is optional and mainly for API compatibility.
    model.set_loss(VocabParallelCrossEntropyLoss, adapter_name=adapter_name)
    model.set_optimizer(AdamW, lr=1e-4, adapter_name=adapter_name)
    model.set_lr_scheduler(LinearLR, adapter_name=adapter_name)

    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name=adapter_name))

    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch, adapter_name=adapter_name)
        if step % 16 == 0:
            logger.info(f'Step {step // 16}, loss: {output}')
        model.clip_grad_norm(1.0, adapter_name=adapter_name)
        model.step(adapter_name=adapter_name)
        model.zero_grad(adapter_name=adapter_name)
        model.lr_step(adapter_name=adapter_name)
        if step % 100 == 0:
            model.save('./output/megatron_lora', adapter_name=adapter_name)
        # Early stop for testing
        if args.max_steps and step >= args.max_steps * 16:
            logger.info(f'Reached max_steps ({args.max_steps}), stopping.')
            break

    logger.info('Training completed!')


def cleanup():
    """Clean up distributed resources."""
    import torch.distributed as dist
    try:
        # Barrier to ensure all processes are synchronized before cleanup
        if dist.is_initialized():
            dist.barrier()
        from megatron.core import parallel_state as mpu
        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    except Exception:
        pass
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        train()
    finally:
        cleanup()
