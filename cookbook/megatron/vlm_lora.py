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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import MegatronCrossEntropyLoss
from twinkle.model import MegatronModel
from twinkle.processor import InputProcessor

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
parser.add_argument('--samples', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
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
    
    # Device mesh: Match Megatron's order "tp-cp-ep-dp-pp"
    device_mesh = DeviceMesh(
        device_type='cuda',
        mesh=np.arange(WORLD_SIZE).reshape(PP_SIZE, DP_SIZE, CP_SIZE, TP_SIZE),
        mesh_dim_names=('pp', 'dp', 'cp', 'tp'),
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
        )
    
    # Create Megatron model
    # Note: MegatronModel will use TwinkleBridgeInitializer which sets up global args
    if args.mode == 'ray':
        model = MegatronModel(
            pretrained_model_name_or_path=args.model,
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=PP_SIZE,
            context_parallel_size=CP_SIZE,
            mixed_precision='bf16',
            recompute_granularity='selective',
            remote_group=GROUP_NAME,
            device_mesh=device_mesh,
        )
    else:
        model = MegatronModel(
            pretrained_model_name_or_path=args.model,
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=PP_SIZE,
            context_parallel_size=CP_SIZE,
            mixed_precision='bf16',
            recompute_granularity='selective',
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
    model.set_template('Qwen3VLTemplate', adapter_name=adapter_name)
    
    # Set up processor for input collation
    model.set_processor(InputProcessor, padding_side='right', adapter_name=adapter_name)
    
    # Set up loss
    model.set_loss(MegatronCrossEntropyLoss, adapter_name=adapter_name)
    
    # Set up optimizer
    model.set_optimizer(AdamW, lr=1e-4, adapter_name=adapter_name)
    model.set_lr_scheduler(LinearLR, adapter_name=adapter_name)
    
    logger.info(model.get_train_configs(adapter_name=adapter_name))
    
    # Training loop
    losses = []
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch, adapter_name=adapter_name)
        
        if step % GAS == 0:
            avg_loss = float(output) if output is not None else 0.0
            losses.append(avg_loss)
            logger.info(f'Step {step // GAS}, loss: {avg_loss:.4f}')
        
        model.clip_grad_norm(1.0, adapter_name=adapter_name)
        model.step(adapter_name=adapter_name)
        model.zero_grad(adapter_name=adapter_name)
        model.lr_step(adapter_name=adapter_name)
        
        # Early stop
        if args.max_steps and step >= args.max_steps * GAS:
            logger.info(f'Reached max_steps ({args.max_steps}), stopping.')
            break
    
    # Summary
    if losses:
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info(f"Steps: {len(losses)}")
        logger.info(f"Initial loss: {losses[0]:.4f}")
        logger.info(f"Final loss: {losses[-1]:.4f}")
        logger.info(f"Average loss: {np.mean(losses):.4f}")
    
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
