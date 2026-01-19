# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-Core LoRA training example.

Supports both local (torchrun) and Ray execution modes.

Usage (Local mode):
    torchrun --nproc_per_node=4 cookbook/megatron/lora.py --tp_size 2 --pp_size 2

Usage (Ray mode):
    TRUST_REMOTE_CODE=1 python cookbook/megatron/lora.py --mode ray --tp_size 2 --pp_size 2 --num_gpus 4
"""
import argparse
import os

import numpy as np
# CRITICAL: Set CUDA device before any CUDA imports (local mode only)
import torch
from peft import LoraConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

import twinkle
from twinkle import (DeviceGroup, DeviceMesh, Platform, get_device_placement,
                     get_logger)
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import MegatronCrossEntropyLoss
from twinkle.model import MegatronModel
from twinkle.processor import InputProcessor

# Parse arguments first to determine mode
parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    default='local',
                    choices=['local', 'ray'])
parser.add_argument('--tp_size', type=int, default=1)
parser.add_argument('--pp_size', type=int, default=1)
parser.add_argument('--cp_size', type=int, default=1)
parser.add_argument('--num_gpus',
                    type=int,
                    default=4,
                    help='Number of GPUs (Ray mode only)')
parser.add_argument('--max_steps', type=int, default=None)
parser.add_argument('--model',
                    type=str,
                    default='ms://Qwen/Qwen2.5-7B-Instruct')
GAS = 16 # gradient accumulation steps
args = parser.parse_args()

# Set mode in environment before importing twinkle
os.environ['TWINKLE_MODE'] = args.mode

if args.mode == 'local':
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(LOCAL_RANK)

logger = get_logger()


def create_dataset():
    dataset = Dataset(
        dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template',
                         model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True, load_from_cache_file=False)
    return dataset


def train():
    # Get parallelism config
    TP_SIZE = args.tp_size
    PP_SIZE = args.pp_size
    CP_SIZE = args.cp_size

    if args.mode == 'local':
        WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
    else:
        WORLD_SIZE = args.num_gpus

    DP_SIZE = WORLD_SIZE // (TP_SIZE * PP_SIZE * CP_SIZE)

    # Device mesh: Match Megatron's order "tp-cp-ep-dp-pp" from innermost to outermost
    device_mesh = DeviceMesh(
        device_type='cuda',
        mesh=np.arange(WORLD_SIZE).reshape(PP_SIZE, DP_SIZE, CP_SIZE, TP_SIZE),
        mesh_dim_names=('pp', 'dp', 'cp', 'tp'),
    )

    # Device group name - used as remote_group in Ray mode
    GROUP_NAME = 'model'

    device_group = [
        DeviceGroup(
            name=GROUP_NAME,
            ranks=list(range(WORLD_SIZE)),
            device_type=Platform.get_platform().device_prefix(),
        )
    ]

    twinkle.initialize(
        mode=args.mode,
        nproc_per_node=WORLD_SIZE,
        groups=device_group,
        global_device_mesh=device_mesh,
        lazy_collect=False,
    )

    # Use smaller batch size for single GPU to avoid OOM
    batch_size = 2 if WORLD_SIZE == 1 else 8

    # In Ray mode, pass remote_group and device_mesh
    if args.mode == 'ray':
        dataloader = DataLoader(
            dataset=create_dataset,
            batch_size=batch_size,
            remote_group=GROUP_NAME,
            device_mesh=device_mesh,
        )
        model = MegatronModel(
            pretrained_model_name_or_path=args.model,
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=PP_SIZE,
            context_parallel_size=CP_SIZE,
            mixed_precision='bf16',
            recompute_granularity='full' if WORLD_SIZE <= 2 else 'selective',
            remote_group=GROUP_NAME,
            device_mesh=device_mesh,
        )
    else:
        dataloader = DataLoader(dataset=create_dataset, batch_size=batch_size)
        model = MegatronModel(
            pretrained_model_name_or_path=args.model,
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=PP_SIZE,
            context_parallel_size=CP_SIZE,
            mixed_precision='bf16',
            recompute_granularity='full' if WORLD_SIZE <= 2 else 'selective',
        )

    lora_config = LoraConfig(target_modules='all-linear')
    adapter_name = 'lora'
    model.add_adapter_to_model(adapter_name,
                               lora_config,
                               gradient_accumulation_steps=GAS)
    model.set_template('Qwen3Template', adapter_name=adapter_name)
    model.set_processor(InputProcessor,
                        padding_side='right',
                        adapter_name=adapter_name)
    model.set_loss(MegatronCrossEntropyLoss, adapter_name=adapter_name)
    model.set_optimizer(AdamW, lr=1e-4, adapter_name=adapter_name)
    model.set_lr_scheduler(LinearLR, adapter_name=adapter_name)

    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name=adapter_name))

    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch,
                                        adapter_name=adapter_name)
        if step % GAS == 0:
            logger.info(f'Step {step // 16}, loss: {output}')
        model.clip_grad_norm(1.0, adapter_name=adapter_name)
        model.step(adapter_name=adapter_name)
        model.zero_grad(adapter_name=adapter_name)
        model.lr_step(adapter_name=adapter_name)
        if step > 0 and step % (100 * GAS) == 0:
            model.save('./output/megatron_lora', adapter_name=adapter_name)
        # Early stop for testing
        if args.max_steps and step >= args.max_steps * GAS:
            logger.info(f'Reached max_steps ({args.max_steps}), stopping.')
            break
    model.save('./output/megatron_lora', adapter_name=adapter_name)
    logger.info('Training completed!')

def cleanup():
    """Clean up distributed resources."""
    import torch.distributed as dist
    try:
        if dist.is_initialized():
            dist.barrier()
        from megatron.core import parallel_state as mpu
        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        train()
    finally:
        cleanup()
