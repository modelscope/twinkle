# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron-Core LoRA training example.

Supports both local (torchrun) and Ray execution modes.

Usage (Local mode with 8 GPUs: TP=2, PP=2, DP=2):
    torchrun --nproc_per_node=8 cookbook/megatron/lora.py --tp_size 2 --pp_size 2 --dp_size 2

Usage (Ray mode):
    TRUST_REMOTE_CODE=1 python cookbook/megatron/lora.py --mode ray --tp_size 2 --pp_size 2 --dp_size 2
"""
import argparse
import os

# CRITICAL: Set CUDA device before any CUDA imports (local mode only)
import torch
from peft import LoraConfig

import twinkle
from twinkle import (DeviceGroup, DeviceMesh, Platform, get_device_placement,
                     get_logger)
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.processor import InputProcessor

# Parse arguments first to determine mode
parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    default='local',
                    choices=['local', 'ray'])
parser.add_argument('--max_steps', type=int, default=20)
parser.add_argument('--model',
                    type=str,
                    default='ms://Qwen/Qwen2.5-7B-Instruct')
parser.add_argument('--dp_size', type=int, default=None, help='Data parallel size (default: auto-calculated)')
parser.add_argument('--tp_size', type=int, default=1, help='Tensor parallel size')
parser.add_argument('--pp_size', type=int, default=1, help='Pipeline parallel size')
parser.add_argument('--vpp_size', type=int, default=None, help='Virtual Pipeline Parallel size (default: None)')
parser.add_argument('--cp_size', type=int, default=1, help='Context parallel size')
parser.add_argument('--sequence_parallel', type=lambda x: x.lower() == 'true', default=True, 
                    help='Enable sequence parallel (default: True)')
parser.add_argument('--micro_batch_size', type=int, default=1, 
                    help='Micro batch size per DP rank (default: 1). For VPP, increase this to >= PP*VPP')
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
    dataset.set_template('Template',
                         model_id=args.model)
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True, load_from_cache_file=False)
    return dataset


def train():
    # Calculate dp_size from world_size and other parallelism dimensions if not specified
    if args.mode == 'local':
        WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
    else:
        WORLD_SIZE = 8  # Default for Ray mode
    
    dp_size = args.dp_size
    if dp_size is None:
        # Auto-calculate dp_size from world_size and other parallelism dimensions
        dp_size = WORLD_SIZE // (args.tp_size * args.pp_size * args.cp_size)
        if dp_size < 1:
            raise ValueError(f"Invalid parallelism config: world_size={WORLD_SIZE}, "
                           f"tp_size={args.tp_size}, pp_size={args.pp_size}, cp_size={args.cp_size}. "
                           f"Total parallelism ({args.tp_size * args.pp_size * args.cp_size}) exceeds world_size.")
    
    # Validate total parallelism matches world_size
    total_parallelism = dp_size * args.tp_size * args.pp_size * args.cp_size
    if total_parallelism != WORLD_SIZE:
        raise ValueError(f"Total parallelism ({total_parallelism}) != world_size ({WORLD_SIZE}). "
                        f"dp={dp_size}, tp={args.tp_size}, pp={args.pp_size}, cp={args.cp_size}")
    
    # Use DeviceMesh.from_sizes for flexible parallelism configuration
    device_mesh = DeviceMesh.from_sizes(
        dp_size=dp_size, 
        pp_size=args.pp_size,
        tp_size=args.tp_size, 
        cp_size=args.cp_size,
        vpp_size=args.vpp_size,
    )

    # Device group name - used as remote_group in Ray mode
    GROUP_NAME = 'model'

    device_group = [
        DeviceGroup(
            name=GROUP_NAME,
            ranks=device_mesh.world_size,
            device_type=Platform.get_platform().device_prefix(),
        )
    ]

    twinkle.initialize(
        mode=args.mode,
        nproc_per_node=device_mesh.world_size,
        groups=device_group,
        global_device_mesh=device_mesh,
        lazy_collect=args.mode == 'ray',
    )

    # For VPP, num_microbatches must be >= PP * VPP
    # micro_batch_size is per forward step, batch_size is total per optimizer step
    micro_batch_size = args.micro_batch_size
    batch_size = micro_batch_size * device_mesh.data_world_size

    # In Ray mode, pass remote_group
    _remote_args = {'device_mesh': device_mesh}
    if args.mode == 'ray':
        _remote_args['remote_group'] = GROUP_NAME

    dataloader = DataLoader(dataset=create_dataset, batch_size=batch_size, **_remote_args)
    model = MegatronModel(
        model_id=args.model,
        sequence_parallel=args.sequence_parallel,
        mixed_precision='bf16',
        recompute_granularity='selective',
        **_remote_args
    )

    lora_config = LoraConfig(target_modules='all-linear')
    adapter_name = 'lora'
    model.add_adapter_to_model(adapter_name,
                               lora_config,
                               gradient_accumulation_steps=GAS)
    model.set_template('Template', model_id=args.model, adapter_name=adapter_name)
    model.set_processor(InputProcessor,
                        padding_side='right',
                        adapter_name=adapter_name)
    model.set_optimizer('default', lr=1e-4, adapter_name=adapter_name)
    model.set_lr_scheduler('default', lr_decay_steps=1000, max_lr=1e-4, adapter_name=adapter_name)
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name=adapter_name))

    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch,
                                        adapter_name=adapter_name)
        if step % GAS == 0:
            loss_value = output() if callable(output) else output
            logger.info(f'Step {step // GAS}, loss: {loss_value}')
        model.clip_grad_norm(1.0, adapter_name=adapter_name)
        model.step(adapter_name=adapter_name)
        model.zero_grad(adapter_name=adapter_name)
        model.lr_step(adapter_name=adapter_name)
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
