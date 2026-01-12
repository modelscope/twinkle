# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-Core LoRA training example with full 4D parallelism.

This example demonstrates LoRA fine-tuning using Megatron-Core backend.
Supports Tensor Parallel (TP), Pipeline Parallel (PP), Context Parallel (CP),
and Data Parallel (DP). DP is automatically calculated from WORLD_SIZE.

The script uses Megatron's get_forward_backward_func() for unified pipeline
scheduling, ensuring proper multi-tenant isolation through process groups.

TODO: Add Expert Parallel (EP) support for MoE models.

Usage (8 GPUs with CP2 PP2 TP2, DP auto-calculated as 1):
    torchrun --nproc_per_node=8 cookbook/megatron/lora.py \
        --tp_size 2 --pp_size 2 --cp_size 2

Usage (4 GPUs with TP2, DP auto-calculated as 2):
    torchrun --nproc_per_node=4 cookbook/megatron/lora.py --tp_size 2

Usage (single GPU for debugging):
    torchrun --nproc_per_node=1 cookbook/megatron/lora.py

Note: WORLD_SIZE is automatically detected from torchrun, no need to specify it twice.
"""
import argparse
import os

import numpy as np
from peft import LoraConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

import twinkle
from twinkle import get_device_placement, get_logger, DeviceMesh, DeviceGroup, Platform
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import MegatronCrossEntropyLoss
from twinkle.model import MegatronModel
from twinkle.processor import InputProcessor

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Megatron LoRA Training with 4D Parallelism')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='local',
                        choices=['local', 'ray'],
                        help='Distributed mode: local (torchrun) or ray')
    
    # Number of GPUs
    parser.add_argument('--nproc_per_node', type=int, default=4,
                        help='Total number of GPUs')
    
    # 4D Parallelism configuration
    # Total GPUs = DP * CP * PP * TP (DP is auto-calculated)
    # TODO: Add EP (Expert Parallel) for MoE models
    parser.add_argument('--tp_size', type=int, default=1,
                        help='Tensor Parallel size (splits model layers horizontally)')
    parser.add_argument('--pp_size', type=int, default=1,
                        help='Pipeline Parallel size (splits model layers vertically)')
    parser.add_argument('--cp_size', type=int, default=1,
                        help='Context Parallel size (splits sequence across GPUs)')
    # Note: DP size is automatically calculated as: WORLD_SIZE / (TP * PP * CP)
    
    # Sequence parallel (usually enabled with TP > 1)
    parser.add_argument('--sequence_parallel', action='store_true', default=False,
                        help='Enable sequence parallelism (recommended when TP > 1)')
    
    # Max steps for quick testing
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum training steps (for testing)')
    
    args = parser.parse_args()
    
    # Auto-detect world size from environment (set by torchrun)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.world_size = world_size
    
    # Auto-calculate DP size from total GPUs and model parallel sizes
    model_parallel_size = args.tp_size * args.pp_size * args.cp_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f'WORLD_SIZE ({world_size}) must be divisible by '
            f'TP({args.tp_size}) * PP({args.pp_size}) * CP({args.cp_size}) = {model_parallel_size}'
        )
    args.dp_size = world_size // model_parallel_size
    
    logger.info(f'4D Parallelism config: DP={args.dp_size} (auto), TP={args.tp_size}, '
                f'PP={args.pp_size}, CP={args.cp_size}, Total GPUs={world_size}')
    
    return args


def create_device_mesh(args) -> DeviceMesh:
    """Create device mesh for Megatron 4D parallelism.
    
    Megatron uses the following parallelism hierarchy (outer to inner):
    - Data Parallel (DP): Replicates model, splits data (auto-calculated)
    - Context Parallel (CP): Splits sequence across GPUs  
    - Pipeline Parallel (PP): Splits layers across stages
    - Tensor Parallel (TP): Splits layers horizontally
    
    TODO: Add Expert Parallel (EP) dimension for MoE models.
    
    Mesh shape: (dp, cp, pp, tp)
    """
    total_gpus = args.world_size
    
    # Create mesh with shape (dp, cp, pp, tp)
    mesh = np.arange(total_gpus).reshape(args.dp_size, args.cp_size, args.pp_size, args.tp_size)
    
    device_mesh = DeviceMesh(
        device_type='cuda',
        mesh=mesh,
        mesh_dim_names=('dp', 'cp', 'pp', 'tp'),
    )
    return device_mesh


def create_device_group(args):
    """Create device group for model placement."""
    device_group = [
        DeviceGroup(
            name='model',
            ranks=list(range(args.world_size)),
            device_type=Platform.get_platform().device_prefix(),
        )
    ]
    return device_group


def create_dataset():
    """Create and preprocess dataset."""
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    # IMPORTANT: Use load_from_cache_file=False to avoid stale cache with incorrect labels
    dataset.encode(batched=True, load_from_cache_file=False)
    return dataset


def train(args):
    """Main training function with 4D parallelism support."""
    # Create dataloader
    dataloader = DataLoader(dataset=create_dataset, batch_size=8)
    
    # Create Megatron model with 4D parallelism
    # TODO: Add expert_model_parallel_size for MoE models
    model = MegatronModel(
        pretrained_model_name_or_path='ms://Qwen/Qwen2.5-7B-Instruct',
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        context_parallel_size=args.cp_size,
        sequence_parallel=args.sequence_parallel,
        mixed_precision='bf16',
    )
    
    # Set template, processor, loss on DEFAULT adapter FIRST
    # These will be copied when adding LoRA adapter
    model.set_template('Qwen3Template')
    model.set_processor(InputProcessor, padding_side='right')
    model.set_loss(MegatronCrossEntropyLoss)
    
    # Configure LoRA adapter
    lora_config = LoraConfig(
        target_modules='all-linear'
    )
    
    # Add LoRA adapter - template, processor, loss_instance will be copied from default
    model.add_adapter_to_model('lora', lora_config, gradient_accumulation_steps=16)
    
    # Set optimizer and scheduler for LoRA adapter (must be after add_adapter_to_model)
    model.set_optimizer(AdamW, lr=1e-4, adapter_name='lora')
    model.set_lr_scheduler(LinearLR, adapter_name='lora')
    
    # Print training configuration
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name='lora'))
    
    # Training loop
    gradient_accumulation_steps = 16
    optimizer_step = 0
    max_steps = args.max_steps
    
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch, adapter_name='lora')
        
        # Only perform optimizer step at gradient accumulation boundary
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer_step = (step + 1) // gradient_accumulation_steps
            
            # Log loss
            logger.info(f'Current is step {optimizer_step}, loss: {output}')
            
            # Gradient clipping and optimizer step
            model.clip_grad_norm(1.0, adapter_name='lora')
            model.step(adapter_name='lora')
            model.zero_grad(adapter_name='lora')
            model.lr_step(adapter_name='lora')
            
            # Save checkpoint every 100 optimizer steps
            if optimizer_step % 100 == 0:
                model.save('./output/megatron_lora', adapter_name='lora')
            
            # Check max_steps for early stopping (for testing)
            if max_steps is not None and optimizer_step >= max_steps:
                logger.info(f'Reached max_steps ({max_steps}), stopping training.')
                break
    
    # Save final checkpoint
    logger.info(f'Training completed! Final step: {optimizer_step}')
    model.save('./output/megatron_lora', adapter_name='lora')


def main():
    args = parse_args()
    
    # Set TWINKLE_MODE environment variable for strategy to detect
    os.environ['TWINKLE_MODE'] = args.mode
    
    # Create device mesh and group
    device_mesh = create_device_mesh(args)
    device_group = create_device_group(args)
    
    # Initialize twinkle with specified mode
    twinkle.initialize(
        mode=args.mode,
        nproc_per_node=args.world_size,
        groups=device_group,
        global_device_mesh=device_mesh,
        lazy_collect=False,
    )
    
    try:
        # Start training
        train(args)
    finally:
        # Clean up distributed process groups
        cleanup_distributed()


def cleanup_distributed():
    """Clean up all distributed process groups."""
    import torch
    import torch.distributed as dist
    
    # Synchronize all processes before cleanup
    if dist.is_initialized():
        try:
            # Use barrier with timeout to prevent hanging
            dist.barrier()
        except Exception as e:
            logger.warning(f'Barrier failed during cleanup: {e}')
        
        try:
            dist.destroy_process_group()
        except Exception as e:
            logger.warning(f'Failed to destroy process group: {e}')
    
    # Also clean up Megatron's parallel state if initialized
    try:
        from megatron.core import parallel_state as mpu
        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    except Exception:
        pass
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
