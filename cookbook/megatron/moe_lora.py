# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron-Core MoE (Mixture of Experts) LoRA training example.

Supports Expert Parallel (EP) training in both local (torchrun) and Ray modes.

Usage (Local mode with EP=2):
    torchrun --nproc_per_node=4 cookbook/megatron/moe_lora.py --tp_size 2 --pp_size 1 --ep_size 2

Usage (Ray mode with EP=2):
    TRUST_REMOTE_CODE=1 python cookbook/megatron/moe_lora.py --mode ray --tp_size 2 --pp_size 1 --ep_size 2 --num_gpus 4
"""
import argparse
import os

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

GAS = 16 # gradient accumulation steps
# Parse arguments first to determine mode

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    default='local',
                    choices=['local', 'ray'])
parser.add_argument('--dp_size', type=int, default=2)
parser.add_argument('--tp_size', type=int, default=2)
parser.add_argument('--pp_size', type=int, default=2)
parser.add_argument('--vpp_size', type=int, default=2)
parser.add_argument('--cp_size', type=int, default=2)
parser.add_argument('--ep_size',
                    type=int,
                    default=2,
                    help='Expert parallel size')
parser.add_argument('--max_steps', type=int, default=100)
parser.add_argument(
    '--model',
    type=str,
    default='ms://Qwen/Qwen3-30B-A3B',
    help='MoE model path. Default: Qwen3-30B-A3B (128 experts)')
parser.add_argument(
    '--sequence_parallel',
    action='store_true',
    default=True,
    help='Enable sequence parallel (auto-enabled for MoE with TP > 1)')
args, unknown = parser.parse_known_args()

logger = get_logger()


def create_dataset():
    """Create dataset for MoE training."""
    dataset = Dataset(
        dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    # Use Qwen3 template for MoE model
    dataset.set_template('Template', model_id=args.model)
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True, load_from_cache_file=False)
    return dataset


def train():
    device_mesh = DeviceMesh.from_sizes(dp_size=args.dp_size, pp_size=args.pp_size,
                                        tp_size=args.tp_size, cp_size=args.cp_size, ep_size=args.ep_size,
                                        vpp_size=args.vpp_size)

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
        groups=device_group,
        global_device_mesh=device_mesh,
        lazy_collect=True,
    )

    # Smaller batch size for MoE models (larger memory footprint)
    batch_size = 4

    _remote_args = {}
    if args.mode == 'ray':
        _remote_args = {
            'remote_group': GROUP_NAME,
            'device_mesh': device_mesh,
        }

    dataloader = DataLoader(dataset=create_dataset, batch_size=batch_size, **_remote_args)
    model = MegatronModel(
        model_id=args.model,
        sequence_parallel=args.sequence_parallel,
        mixed_precision='bf16',
        recompute_granularity='selective',
        **_remote_args
    )

    # LoRA config - target all linear layers in MoE (including experts)
    lora_config = LoraConfig(
        target_modules='all-linear',
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    adapter_name = 'lora'
    model.add_adapter_to_model(adapter_name, lora_config)
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name=adapter_name))

    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch,
                                        micro_batch_size=1,
                                        adapter_name=adapter_name)
        if step % GAS == 0:
            logger.info(f'Step {step // GAS}, loss: {output}')
        model.clip_grad_and_step()
        if step > 0 and step % (100 * GAS) == 0:
            model.save('./output/megatron_moe_lora', adapter_name=adapter_name)
        # Early stop for testing
        if args.max_steps and step >= args.max_steps * GAS:
            logger.info(f'Reached max_steps ({args.max_steps}), stopping.')
            break
    model.save('./output/megatron_moe_lora', adapter_name=adapter_name)
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
