# Copyright (c) twinkle authors. All rights reserved.
"""VLM (Vision-Language Model) LoRA training example with Transformers backend.

This example demonstrates training Qwen3-VL with LoRA using the 
HuggingFace Transformers backend (not Megatron).

Usage:
    torchrun --nproc_per_node=2 cookbook/sft/vlm_lora.py
"""
import argparse
import io
import os
import time
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig

import twinkle
from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.loss import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
parser.add_argument('--dataset', type=str, default='ms://AI-ModelScope/LaTeX_OCR')
parser.add_argument('--subset', type=str, default='human_handwrite')
parser.add_argument('--samples', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--max_steps', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lora_rank', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)  # swift uses 0.05 by default
parser.add_argument('--max_grad_norm', type=float, default=1.0)
args = parser.parse_args()

# Initialize twinkle
twinkle.initialize(mode='local')

logger = get_logger()

GAS = 4  # gradient accumulation steps


def preprocess_latex_ocr(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LaTeX_OCR sample to VLM format."""
    image_data = sample.get('image')
    formula = sample.get('text', sample.get('formula', ''))
    
    if image_data is None or not formula:
        return {'messages': [], 'images': []}
    
    # Convert image data to PIL Image
    if isinstance(image_data, dict) and 'bytes' in image_data:
        img = Image.open(io.BytesIO(image_data['bytes']))
    elif isinstance(image_data, Image.Image):
        img = image_data
    elif isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    else:
        return {'messages': [], 'images': []}
    
    # Create conversation with image placeholder
    messages = [
        {'role': 'user', 'content': '<image>\nUsing LaTeX to perform OCR on the image.'},
        {'role': 'assistant', 'content': formula}
    ]
    
    return {'messages': messages, 'images': [img]}


def create_dataset():
    """Create VLM dataset with preprocessing."""
    dataset = Dataset(
        dataset_meta=DatasetMeta(
            dataset_id=args.dataset,
            subset_name=args.subset,
            split='train',
            data_slice=range(args.samples)
        )
    )
    
    # Apply preprocessing
    dataset.dataset = dataset.dataset.map(
        preprocess_latex_ocr,
        batched=False,
        load_from_cache_file=False
    )
    
    # Filter out invalid samples
    dataset.dataset = dataset.dataset.filter(
        lambda x: len(x.get('messages', [])) > 0
    )
    
    # Set up Qwen3-VL template and encoding
    dataset.set_template('Qwen3VLTemplate', model_id=args.model)
    dataset.encode(batched=False, load_from_cache_file=False)
    
    return dataset


def train():
    """Main training function."""
    logger.info(f"VLM LoRA Training (Transformers backend)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}/{args.subset} ({args.samples} samples)")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation steps: {GAS}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    logger.info(f"Max grad norm: {args.max_grad_norm}")
    logger.info("=" * 60)
    
    # Create dataloader
    if args.batch_size is None:
        args.batch_size = int(os.environ.get('WORLD_SIZE', 4))
    dataloader = DataLoader(
        dataset=create_dataset,
        batch_size=args.batch_size,
    )
    from modelscope import Qwen3VLForConditionalGeneration
    
    model = TransformersModel(
        model_cls=Qwen3VLForConditionalGeneration,
        model_id=args.model,
        mixed_precision='bf16',
        trust_remote_code=True,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    model.add_adapter_to_model(
        'vlm_lora',
        lora_config,
        gradient_accumulation_steps=GAS
    )
    
    # Set up template for VLM
    model.set_template('Qwen3VLTemplate', model_id=args.model)
    
    # Set up processor for input collation
    model.set_processor(InputProcessor, padding_side='right')
    
    # Set up loss
    model.set_loss(CrossEntropyLoss)
    
    # Set up optimizer
    model.set_optimizer(AdamW, lr=args.lr)
    model.set_lr_scheduler(LinearLR)
    
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    
    # Training loop with detailed logging
    losses = []
    grad_norms = []
    step_times = []
    total_start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        step_start_time = time.time()
        
        loss = model.forward_backward(inputs=batch)
        grad_norm = model.clip_grad_and_step(max_grad_norm=args.max_grad_norm)
        
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        step_times.append(step_time)
        
        # Log every GAS steps (when optimizer actually steps)
        if (step + 1) % GAS == 0:
            optimizer_step = (step + 1) // GAS
            loss_value = loss() if callable(loss) else loss
            avg_loss = float(loss_value) if loss_value is not None else 0.0
            losses.append(avg_loss)
            
            # Get grad_norm (might be None if not synced)
            grad_norm_value = float(grad_norm) if grad_norm is not None else 0.0
            grad_norms.append(grad_norm_value)
            
            avg_step_time = np.mean(step_times[-GAS:])
            
            logger.info(f'Step {optimizer_step:3d} | loss: {avg_loss:.4f} | grad_norm: {grad_norm_value:.4f} | time: {avg_step_time:.3f}s')
        
        # Early stop based on optimizer steps
        if args.max_steps and (step + 1) >= args.max_steps * GAS:
            break
    
    total_time = time.time() - total_start_time
    
    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total optimizer steps: {len(losses)}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Avg time per optimizer step: {total_time / len(losses) if losses else 0:.3f}s")
    logger.info(f"Initial loss: {losses[0] if losses else 'N/A':.4f}")
    logger.info(f"Final loss: {losses[-1] if losses else 'N/A':.4f}")
    logger.info(f"Avg grad_norm: {np.mean(grad_norms) if grad_norms else 'N/A':.4f}")
    logger.info(f"Min grad_norm: {np.min(grad_norms) if grad_norms else 'N/A':.4f}")
    logger.info(f"Max grad_norm: {np.max(grad_norms) if grad_norms else 'N/A':.4f}")
    logger.info("=" * 60)
    
    # Save model
    output_dir = './output/transformers_vlm_lora'
    model.save(output_dir)
    logger.info(f'Model saved to {output_dir}')
    logger.info('VLM LoRA training completed!')


if __name__ == '__main__':
    train()
