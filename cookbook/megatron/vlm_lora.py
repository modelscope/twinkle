# Copyright (c) twinkle authors. All rights reserved.
# TODO: test

"""Megatron-Core VLM (Vision-Language Model) LoRA training example.

This example demonstrates training Qwen3-VL-8B-Instruct with LoRA using 
Megatron-Core backend for efficient multi-GPU training.

Usage (Local mode):
    torchrun --nproc_per_node=2 cookbook/megatron/vlm_lora.py --tp_size 2

Usage (with custom model):
    torchrun --nproc_per_node=4 cookbook/megatron/vlm_lora.py \
        --tp_size 2 --pp_size 2 \
        --model /path/to/Qwen3-VL-8B-Instruct
"""
import argparse
import io
import os
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_logger, get_device_placement
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.processor import InputProcessor
from twinkle.utils.platform import is_last_rank

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='local', choices=['local', 'ray'])
parser.add_argument('--tp_size', type=int, default=1)
parser.add_argument('--pp_size', type=int, default=1)
parser.add_argument('--cp_size', type=int, default=1)
parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs (Ray mode)')
parser.add_argument('--max_steps', type=int, default=10)
parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
parser.add_argument('--dataset', type=str, default='ms://AI-ModelScope/LaTeX_OCR')
parser.add_argument('--subset', type=str, default='human_handwrite')
parser.add_argument('--samples', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=4)
GAS = 4  # gradient accumulation steps
args = parser.parse_args()

# Set mode before importing twinkle
os.environ['TWINKLE_MODE'] = args.mode

if args.mode == 'local':
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(LOCAL_RANK)

logger = get_logger()


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
    # Note: For VLM models, the template handles image token insertion
    dataset.set_template('Qwen3VLTemplate', model_id=args.model)
    dataset.encode(batched=False, load_from_cache_file=False)
    
    return dataset


def train():
    """Main training function."""
    TP_SIZE = args.tp_size
    PP_SIZE = args.pp_size
    CP_SIZE = args.cp_size
    
    if args.mode == 'local':
        WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
    else:
        WORLD_SIZE = args.num_gpus
    
    DP_SIZE = WORLD_SIZE // (TP_SIZE * PP_SIZE * CP_SIZE)
    
    logger.info(f"VLM LoRA Training: TP={TP_SIZE}, PP={PP_SIZE}, CP={CP_SIZE}, DP={DP_SIZE}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}/{args.subset} ({args.samples} samples)")
    
    # Device mesh: Use DeviceMesh.from_sizes for proper configuration
    device_mesh = DeviceMesh.from_sizes(
        dp_size=DP_SIZE,
        pp_size=PP_SIZE,
        tp_size=TP_SIZE,
        cp_size=CP_SIZE,
    )
    
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
    
    # Create dataloader
    if args.mode == 'ray':
        dataloader = DataLoader(
            dataset=create_dataset,
            batch_size=args.batch_size,
            remote_group=GROUP_NAME,
            device_mesh=device_mesh,
        )
    else:
        dataloader = DataLoader(
            dataset=create_dataset,
            batch_size=args.batch_size,
            device_mesh=device_mesh,
        )
    
    # Create Megatron model
    _remote_args = {}
    if args.mode == 'ray':
        _remote_args = {
            'remote_group': GROUP_NAME,
            'device_mesh': device_mesh,
        }
    
    model = MegatronModel(
        model_id=args.model,
        device_mesh=device_mesh,
        mixed_precision='bf16',
        recompute_granularity='selective',
        sequence_parallel=False,  # VLM may have variable seq lengths
        **_remote_args
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        target_modules='all-linear',
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
    )
    adapter_name = 'vlm_lora'
    
    model.add_adapter_to_model(
        adapter_name,
        lora_config,
        gradient_accumulation_steps=GAS
    )
    
    # Set up template for VLM
    # The template handles image token insertion for Qwen3-VL
    model.set_template('Qwen3VLTemplate', model_id=args.model, adapter_name=adapter_name)
    
    # Set up processor for input collation
    model.set_processor(InputProcessor, padding_side='right', adapter_name=adapter_name)
    
    # Set up optimizer (use Megatron's default optimizer)
    model.set_optimizer('default', lr=1e-4, adapter_name=adapter_name)
    model.set_lr_scheduler('default', lr_decay_steps=1000, max_lr=1e-4, adapter_name=adapter_name)
    
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name=adapter_name))
    
    # Training loop
    losses = []
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch, adapter_name=adapter_name)
        
        if step % GAS == 0:
            loss_value = output() if callable(output) else output
            avg_loss = float(loss_value) if loss_value is not None else 0.0
            logger.info(f'Step {step // GAS}, loss: {avg_loss:.4f}')
        
        model.clip_grad_norm(1.0, adapter_name=adapter_name)
        model.step(adapter_name=adapter_name)
        model.zero_grad(adapter_name=adapter_name)
        model.lr_step(adapter_name=adapter_name)
        
        # Early stop
        if args.max_steps and step >= args.max_steps * GAS:
            logger.info(f'Reached max_steps ({args.max_steps}), stopping.')
            break
    

    # Save model
    output_dir = './output/megatron_vlm_lora'
    model.save(output_dir, adapter_name=adapter_name)
    logger.info(f'Model saved to {output_dir}')
    logger.info('VLM LoRA training completed!')


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
