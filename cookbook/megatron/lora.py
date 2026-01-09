# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-Core LoRA training example.

This example demonstrates LoRA fine-tuning using Megatron-Core backend.
Supports both local (DDP) and Ray distributed modes.

Usage (local mode with 4 GPUs):
    torchrun --nproc_per_node=4 cookbook/megatron/lora.py --mode local

Usage (Ray mode):
    python cookbook/megatron/lora.py --mode ray
"""
import argparse

import numpy as np
from peft import LoraConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import twinkle
from twinkle import get_device_placement, get_logger, DeviceMesh, DeviceGroup, Platform
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CrossEntropyLoss, MegatronCrossEntropyLoss
from twinkle.model import MegatronModel
from twinkle.processor import InputProcessor

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Megatron LoRA Training')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='ray',
                        choices=['local', 'ray'],
                        help='Distributed mode: local (DDP) or ray')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='ms://Qwen/Qwen2.5-7B-Instruct',
                        help='HuggingFace model name or path')
    parser.add_argument('--output_dir', type=str, default='./output/megatron_lora',
                        help='Output directory for checkpoints')
    
    # Parallelism arguments
    parser.add_argument('--nproc_per_node', type=int, default=4,
                        help='Number of processes per node')
    parser.add_argument('--tp_size', type=int, default=2,
                        help='Tensor parallel size')
    parser.add_argument('--dp_size', type=int, default=2,
                        help='Data parallel size')
    parser.add_argument('--sequence_parallel', action='store_true',
                        help='Enable sequence parallelism')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision mode')
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--target_modules', type=str, default='all-linear',
                        help='Target modules for LoRA')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum training steps')
    parser.add_argument('--save_steps', type=int, default=50,
                        help='Checkpoint save interval')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ms://modelscope/competition_math',
                        help='Dataset name')
    
    return parser.parse_args()


def create_device_mesh(args) -> DeviceMesh:
    """Create device mesh for Megatron parallelism."""
    # For Megatron: mesh shape is (dp, tp) 
    # dp_size * tp_size = nproc_per_node
    mesh = np.arange(args.nproc_per_node).reshape(args.dp_size, args.tp_size)
    
    device_mesh = DeviceMesh(
        device_type='cuda',
        mesh=mesh,
        mesh_dim_names=('dp', 'tp'),
    )
    return device_mesh


def create_device_group(args):
    """Create device group for model placement."""
    device_group = [
        DeviceGroup(
            name='model',
            ranks=list(range(args.nproc_per_node)),
            device_type=Platform.get_platform().device_prefix(),
        )
    ]
    return device_group


def create_dataset(args):
    """Create and preprocess dataset."""
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset))
    dataset.set_template('Qwen3Template', model_id=args.model_name)
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    return dataset


def train(args):
    """Main training function."""
    # Create dataloader
    dataloader = DataLoader(
        dataset=lambda: create_dataset(args),
        batch_size=args.batch_size,
    )
    
    # Create Megatron model
    model = MegatronModel(
        pretrained_model_name_or_path=args.model_name,
        tensor_model_parallel_size=args.tp_size,
        sequence_parallel=args.sequence_parallel,
        mixed_precision=args.mixed_precision,
    )
    
    # Configure LoRA adapter
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    
    model.add_adapter_to_model(
        'default', 
        lora_config, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Set template and processor
    model.set_template('Qwen3Template')
    model.set_processor(InputProcessor, padding_side='right')
    
    # Set loss, optimizer, scheduler
    model.set_loss(MegatronCrossEntropyLoss)
    model.set_optimizer(AdamW, lr=args.learning_rate, weight_decay=0.01)
    model.set_lr_scheduler(CosineAnnealingLR, T_max=args.max_steps)
    
    # Print training configuration
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    
    # Training loop
    global_step = 0
    for step, batch in enumerate(dataloader):
        if global_step >= args.max_steps:
            break
        
        # Forward-backward pass
        output = model.forward_backward(inputs=batch)
        
        # Log loss at gradient accumulation boundary
        if step % args.gradient_accumulation_steps == 0:
            logger.info(f'Step {global_step}, Loss: {output}')
            global_step += 1
        
        # Gradient clipping and optimizer step
        model.clip_grad_norm(args.max_grad_norm)
        model.step()
        model.zero_grad()
        model.lr_step()
        
        # Save checkpoint
        if global_step > 0 and global_step % args.save_steps == 0:
            model.save(f'{args.output_dir}/checkpoint-{global_step}')
    
    # Save final model
    model.save(args.output_dir)
    logger.info(f'Model saved to {args.output_dir}')


def main():
    args = parse_args()
    
    # Create device mesh and group
    device_mesh = create_device_mesh(args)
    device_group = create_device_group(args)
    
    # Initialize twinkle with specified mode
    twinkle.initialize(
        mode=args.mode,
        nproc_per_node=args.nproc_per_node,
        groups=device_group,
        global_device_mesh=device_mesh,
        lazy_collect=False,
    )
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()
