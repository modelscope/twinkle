#!/usr/bin/env python
# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-Core LoRA training in Ray mode.

This script uses MegatronWorkerGroup for Ray-based distributed training
with proper Megatron collective operations support.

NOTE: PP > 1 is REQUIRED for training. PP=1 has known gradient flow issues
with PEFT/LoRA and Megatron's forward_backward_no_pipelining.

Usage:
    # TP=2, PP=2 (4 GPUs) - RECOMMENDED
    python cookbook/megatron/lora_ray.py --tp_size 2 --pp_size 2 --num_gpus 4

    # PP=4, TP=1 (4 GPUs)
    python cookbook/megatron/lora_ray.py --tp_size 1 --pp_size 4 --num_gpus 4
    
    # PP=2, TP=1 (2 GPUs)
    python cookbook/megatron/lora_ray.py --tp_size 1 --pp_size 2 --num_gpus 2
"""
import argparse
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
megatron_path = os.environ.get('MEGATRON_LM_PATH', '/mnt/nas2/hujinghan.hjh/Megatron-LM')
sys.path.insert(0, megatron_path)

import ray
import torch
import numpy as np

from twinkle import get_logger
from twinkle.megatron.worker import MegatronWorkerGroup

logger = get_logger()


def create_dataset():
    """Create and prepare the training dataset - same as local mode."""
    from twinkle.dataset import Dataset, DatasetMeta
    
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True, load_from_cache_file=False)
    return dataset


def collate_batch(samples, batch_size: int, max_seq_len: int = 512):
    """Collate samples into a batch with padding."""
    # Take batch_size samples
    samples = samples[:batch_size]
    
    # Get max length in batch (capped at max_seq_len)
    max_len = min(max(len(s['input_ids']) for s in samples), max_seq_len)
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for s in samples:
        ids = s['input_ids'][:max_len]
        pad_len = max_len - len(ids)
        
        input_ids_list.append(ids + [0] * pad_len)
        attention_mask_list.append([1] * len(ids) + [0] * pad_len)
        
        # Labels: use -100 for padding
        labels = s.get('labels', ids)[:max_len]
        labels_list.append(labels + [-100] * pad_len)
    
    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
        'labels': torch.tensor(labels_list, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser(description='Megatron LoRA training in Ray mode')
    parser.add_argument('--tp_size', type=int, default=2, help='Tensor parallel size')
    parser.add_argument('--pp_size', type=int, default=2, help='Pipeline parallel size (must be > 1 for training)')
    parser.add_argument('--cp_size', type=int, default=1, help='Context parallel size')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--max_steps', type=int, default=10, help='Max training steps')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per step')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='ms://Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model path or ID')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    args = parser.parse_args()
    
    # Validate parallelism config
    expected_gpus = args.tp_size * args.pp_size * args.cp_size
    if args.num_gpus < expected_gpus:
        logger.error(f"Need at least {expected_gpus} GPUs for TP={args.tp_size}, "
                     f"PP={args.pp_size}, CP={args.cp_size}, but only {args.num_gpus} provided")
        return 1
    
    # Prepare dataset first (on driver, before Ray workers)
    logger.info("Preparing dataset...")
    dataset = create_dataset()
    samples = [dataset[i] for i in range(min(len(dataset), args.max_steps * args.batch_size + 100))]
    logger.info(f"Loaded {len(samples)} samples")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    logger.info(f"Ray initialized with {args.num_gpus} GPUs")
    logger.info(f"Config: TP={args.tp_size}, PP={args.pp_size}, CP={args.cp_size}")
    
    # Create worker group
    worker_group = MegatronWorkerGroup(
        world_size=args.num_gpus,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        cp_size=args.cp_size,
    )
    
    try:
        # Initialize workers
        logger.info("Initializing workers...")
        results = worker_group.init_all()
        if not all(results):
            raise RuntimeError("Worker initialization failed")
        
        # Create model
        logger.info(f"Loading model: {args.model}")
        results = worker_group.create_model_all(
            pretrained_model_name_or_path=args.model,
            mixed_precision='bf16',
            recompute_granularity='full',
        )
        if not all(results):
            raise RuntimeError("Model creation failed")
        
        # Add LoRA with Megatron layer names
        logger.info("Adding LoRA adapters...")
        lora_config = {
            'target_modules': ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'],
            'r': args.lora_r,
            'lora_alpha': args.lora_r,
            'lora_dropout': 0.0,
        }
        results = worker_group.add_lora_all(lora_config)
        if not all(results):
            raise RuntimeError("LoRA addition failed")
        
        # Set optimizer
        logger.info(f"Setting optimizer with lr={args.lr}")
        results = worker_group.set_optimizer_all(lr=args.lr)
        if not all(results):
            raise RuntimeError("Optimizer setup failed")
        
        # Training loop
        logger.info(f"Starting training for {args.max_steps} steps...")
        losses = []
        
        # Use same batch for all steps to verify loss decreases (overfitting test)
        fixed_batch = collate_batch(samples[:args.batch_size], args.batch_size, args.max_seq_len)
        
        for step in range(args.max_steps):
            batch = fixed_batch
            
            # Forward-backward
            step_losses = worker_group.forward_backward_all(batch)
            
            # Get valid loss (non-zero from last PP stage)
            valid_losses = [l for l in step_losses if l > 0]
            avg_loss = np.mean(valid_losses) if valid_losses else 0.0
            losses.append(avg_loss)
            
            logger.info(f"Step {step:3d}/{args.max_steps}, loss: {avg_loss:.4f}")
            
            # Optimizer step
            worker_group.step_all()
        
        # Check loss trend
        logger.info("=" * 60)
        logger.info("Training Summary:")
        logger.info(f"  Initial loss: {losses[0]:.4f}")
        logger.info(f"  Final loss:   {losses[-1]:.4f}")
        logger.info(f"  Loss change:  {losses[-1] - losses[0]:.4f}")
        
        # Validation checks (aligned with local mode expectations)
        initial_ok = losses[0] < 3  # Real data should have initial loss < 3
        decreasing = losses[-1] < losses[0]  # Should decrease over training
        
        if initial_ok:
            logger.info("✓ Initial loss is reasonable (< 3)")
        else:
            logger.warning(f"✗ Initial loss {losses[0]:.4f} is too high (expected < 3)")
            
        if decreasing:
            logger.info("✓ Loss is decreasing (training is working)")
        else:
            logger.warning("✗ Loss is not decreasing")
        
        logger.info("=" * 60)
        logger.info("Training completed!")
        
        return 0 if (initial_ok and decreasing) else 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        logger.info("Cleaning up...")
        try:
            worker_group.cleanup_all()
        except Exception:
            pass
        try:
            worker_group.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
