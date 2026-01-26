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
from typing import Any, Dict

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
parser.add_argument('--samples', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
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
    
    # Create dataloader
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
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
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
    
    # Training loop
    # Note: gradient accumulation is handled internally by the framework.
    # clip_grad_and_step() internally checks cur_step % gradient_accumulation_steps
    # and only performs the actual optimizer step when conditions are met.
    for step, batch in enumerate(dataloader):
        loss = model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        
        # Log every GAS steps (when optimizer actually steps)
        if (step + 1) % GAS == 0:
            optimizer_step = (step + 1) // GAS
            loss_value = loss() if callable(loss) else loss
            logger.info(f'Optimizer step {optimizer_step}, loss: {loss_value:.4f}')
        
        # Early stop based on optimizer steps
        if args.max_steps and (step + 1) >= args.max_steps * GAS:
            break
    
    # Save model
    output_dir = './output/transformers_vlm_lora'
    model.save(output_dir)
    logger.info(f'Model saved to {output_dir}')
    logger.info('VLM LoRA training completed!')


if __name__ == '__main__':
    train()
