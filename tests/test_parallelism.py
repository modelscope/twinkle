#!/usr/bin/env python
# Copyright (c) twinkle authors. All rights reserved.
"""Test different parallelism strategies for Megatron backend in local mode.

This script tests various parallelism configurations:
- TP (Tensor Parallel)
- PP (Pipeline Parallel)
- DP (Data Parallel)
- CP (Context Parallel)
- SP (Sequence Parallel, enabled when TP > 1)
- Combined configurations

Uses Qwen2.5-0.5B-Instruct for faster testing.

Usage:
    python cookbook/megatron/test_parallelism.py
"""
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


@dataclass
class TestConfig:
    """Test configuration for a parallelism strategy."""
    name: str
    tp_size: int
    pp_size: int
    cp_size: int = 1
    sp_enabled: bool = False  # Sequence Parallel
    num_gpus: int = 0  # 0 means auto-calculate
    max_steps: int = 2
    expected_to_pass: bool = True
    notes: str = ""
    
    def __post_init__(self):
        if self.num_gpus == 0:
            self.num_gpus = self.tp_size * self.pp_size * self.cp_size
            # Ensure at least 1 for DP
            if self.num_gpus == 0:
                self.num_gpus = 1


# Test configurations
TEST_CONFIGS: List[TestConfig] = [
    # Basic single-GPU
    TestConfig(
        name="Single GPU (TP=1, PP=1)",
        tp_size=1, pp_size=1, cp_size=1,
        num_gpus=1,
        notes="Baseline test"
    ),
    
    # Tensor Parallel only
    TestConfig(
        name="TP=2 (Tensor Parallel)",
        tp_size=2, pp_size=1, cp_size=1,
        notes="Tests tensor sharding"
    ),
    TestConfig(
        name="TP=4 (Tensor Parallel)",
        tp_size=4, pp_size=1, cp_size=1,
        notes="Larger TP"
    ),
    
    # Pipeline Parallel only
    TestConfig(
        name="PP=2 (Pipeline Parallel)",
        tp_size=1, pp_size=2, cp_size=1,
        notes="Tests pipeline stages"
    ),
    TestConfig(
        name="PP=4 (Pipeline Parallel)",
        tp_size=1, pp_size=4, cp_size=1,
        notes="More pipeline stages"
    ),
    
    # TP + PP combinations
    TestConfig(
        name="TP=2, PP=2",
        tp_size=2, pp_size=2, cp_size=1,
        notes="Combined TP+PP"
    ),
    TestConfig(
        name="TP=2, PP=4",
        tp_size=2, pp_size=4, cp_size=1,
        num_gpus=8,
        notes="8-GPU TP+PP"
    ),
    
    # Data Parallel (DP > 1)
    TestConfig(
        name="TP=2, PP=2, DP=2 (8 GPUs)",
        tp_size=2, pp_size=2, cp_size=1,
        num_gpus=8,
        expected_to_pass=False,
        notes="Known issue: P2P deadlock with DP > 1"
    ),
    
    # Context Parallel
    TestConfig(
        name="CP=2 (Context Parallel)",
        tp_size=1, pp_size=1, cp_size=2,
        expected_to_pass=False,
        notes="Known issue: CP communication deadlock"
    ),
    TestConfig(
        name="TP=2, PP=2, CP=2 (8 GPUs)",
        tp_size=2, pp_size=2, cp_size=2,
        num_gpus=8,
        expected_to_pass=False,
        notes="Known issue: CP + PP deadlock"
    ),
    
    # Sequence Parallel (with TP)
    TestConfig(
        name="TP=2 + SP (Sequence Parallel)",
        tp_size=2, pp_size=1, cp_size=1,
        sp_enabled=True,
        notes="SP is typically enabled with TP"
    ),
    TestConfig(
        name="TP=2, PP=2 + SP",
        tp_size=2, pp_size=2, cp_size=1,
        sp_enabled=True,
        notes="Combined TP+PP+SP"
    ),
]


def get_available_gpus() -> int:
    """Get number of available GPUs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except Exception:
        pass
    return 0


def create_test_script() -> str:
    """Create a minimal test script for parallelism testing."""
    script = '''
# Minimal Megatron parallelism test script
import os
import sys
import argparse

# Set CUDA device before any imports
import torch
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))
torch.cuda.set_device(LOCAL_RANK)

import numpy as np
import twinkle
from twinkle import DeviceMesh, DeviceGroup, Platform, get_logger
from twinkle.model import MegatronModel
from peft import LoraConfig
from torch.optim import AdamW

logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--tp_size', type=int, default=1)
parser.add_argument('--pp_size', type=int, default=1)
parser.add_argument('--cp_size', type=int, default=1)
parser.add_argument('--sp_enabled', action='store_true')
parser.add_argument('--max_steps', type=int, default=2)
args = parser.parse_args()

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
TP_SIZE = args.tp_size
PP_SIZE = args.pp_size
CP_SIZE = args.cp_size
DP_SIZE = WORLD_SIZE // (TP_SIZE * PP_SIZE * CP_SIZE)

device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.arange(WORLD_SIZE).reshape(DP_SIZE, CP_SIZE, PP_SIZE, TP_SIZE),
    mesh_dim_names=('dp', 'cp', 'pp', 'tp'),
)

device_group = [
    DeviceGroup(name='model', ranks=list(range(WORLD_SIZE)),
                device_type=Platform.get_platform().device_prefix())
]

twinkle.initialize(
    mode='local',
    nproc_per_node=WORLD_SIZE,
    groups=device_group,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)

# Create model with smaller Qwen2.5-0.5B
model = MegatronModel(
    pretrained_model_name_or_path='ms://Qwen/Qwen2.5-0.5B-Instruct',
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=PP_SIZE,
    context_parallel_size=CP_SIZE,
    sequence_parallel=args.sp_enabled,
    mixed_precision='bf16',
    recompute_granularity='full',
)

# Add LoRA
lora_config = LoraConfig(target_modules='all-linear', r=4)
model.add_adapter_to_model('lora', lora_config, gradient_accumulation_steps=1)
model.set_optimizer(AdamW, lr=1e-4, adapter_name='lora')

logger.info(f"Model initialized: TP={TP_SIZE}, PP={PP_SIZE}, CP={CP_SIZE}, DP={DP_SIZE}, SP={args.sp_enabled}")

# Training loop with dummy data
for step in range(args.max_steps):
    batch = {
        'input_ids': torch.randint(0, 1000, (1, 64), device=f'cuda:{LOCAL_RANK}'),
        'attention_mask': torch.ones(1, 64, device=f'cuda:{LOCAL_RANK}'),
        'labels': torch.randint(0, 1000, (1, 64), device=f'cuda:{LOCAL_RANK}'),
    }
    loss = model.forward_backward(inputs=batch, adapter_name='lora')
    logger.info(f"Step {step}, loss: {loss}")
    model.step(adapter_name='lora')
    model.zero_grad(adapter_name='lora')

logger.info("Training completed successfully!")

# Cleanup
import torch.distributed as dist
if dist.is_initialized():
    dist.barrier()
    from megatron.core import parallel_state as mpu
    if mpu.is_initialized():
        mpu.destroy_model_parallel()
    dist.destroy_process_group()
'''
    return script


def run_test(config: TestConfig, available_gpus: int, test_script_path: str) -> dict:
    """Run a single test configuration."""
    result = {
        'name': config.name,
        'config': f"TP={config.tp_size}, PP={config.pp_size}, CP={config.cp_size}" + 
                  (", SP=True" if config.sp_enabled else ""),
        'gpus': config.num_gpus,
        'status': 'SKIPPED',
        'message': '',
        'duration': 0,
    }
    
    # Check if we have enough GPUs
    if config.num_gpus > available_gpus:
        result['status'] = 'SKIPPED'
        result['message'] = f"Need {config.num_gpus} GPUs, only {available_gpus} available"
        return result
    
    # Build command
    cuda_devices = ','.join(str(i) for i in range(config.num_gpus))
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        '--nproc_per_node', str(config.num_gpus),
        test_script_path,
        '--tp_size', str(config.tp_size),
        '--pp_size', str(config.pp_size),
        '--cp_size', str(config.cp_size),
        '--max_steps', str(config.max_steps),
    ]
    if config.sp_enabled:
        cmd.append('--sp_enabled')
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_devices
    env['MEGATRON_LM_PATH'] = os.environ.get('MEGATRON_LM_PATH', '/mnt/nas2/hujinghan.hjh/Megatron-LM')
    env['PYTHONPATH'] = f"{env['MEGATRON_LM_PATH']}:{os.getcwd()}/src:{env.get('PYTHONPATH', '')}"
    
    # Timeout: 3 minutes per test
    timeout = 180
    
    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout
        )
        duration = time.time() - start_time
        result['duration'] = duration
        
        if proc.returncode == 0:
            # Check if training completed
            if 'Training completed successfully!' in proc.stdout or 'Training completed successfully!' in proc.stderr:
                result['status'] = 'PASSED'
                result['message'] = f"Completed in {duration:.1f}s"
            else:
                result['status'] = 'FAILED'
                result['message'] = "Training did not complete"
        else:
            result['status'] = 'FAILED'
            # Extract error message
            stderr = proc.stderr[-500:] if len(proc.stderr) > 500 else proc.stderr
            result['message'] = f"Exit code {proc.returncode}: {stderr}"
            
    except subprocess.TimeoutExpired:
        result['status'] = 'TIMEOUT'
        result['message'] = f"Exceeded {timeout}s timeout (likely deadlock)"
        result['duration'] = timeout
        # Kill any remaining processes
        subprocess.run(['pkill', '-f', test_script_path], capture_output=True)
        time.sleep(2)
    except Exception as e:
        result['status'] = 'ERROR'
        result['message'] = str(e)
    
    return result


def main():
    print("=" * 80)
    print("Megatron Parallelism Test Suite")
    print("=" * 80)
    
    available_gpus = get_available_gpus()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus == 0:
        print(f"{RED}No GPUs available!{RESET}")
        return 1
    
    # Create test script
    test_script_path = '/tmp/megatron_parallelism_test.py'
    with open(test_script_path, 'w') as f:
        f.write(create_test_script())
    print(f"Test script created: {test_script_path}")
    
    # Run tests
    results = []
    for i, config in enumerate(TEST_CONFIGS):
        print(f"\n[{i+1}/{len(TEST_CONFIGS)}] Testing: {config.name}")
        print(f"    Config: TP={config.tp_size}, PP={config.pp_size}, CP={config.cp_size}, GPUs={config.num_gpus}")
        if config.notes:
            print(f"    Notes: {config.notes}")
        
        result = run_test(config, available_gpus, test_script_path)
        results.append(result)
        
        # Print result
        if result['status'] == 'PASSED':
            status_str = f"{GREEN}PASSED{RESET}"
        elif result['status'] == 'SKIPPED':
            status_str = f"{YELLOW}SKIPPED{RESET}"
        else:
            status_str = f"{RED}{result['status']}{RESET}"
        
        print(f"    Result: {status_str}")
        if result['message']:
            print(f"    Message: {result['message'][:200]}")
        
        # Check if result matches expectation
        if config.expected_to_pass and result['status'] not in ['PASSED', 'SKIPPED']:
            print(f"    {RED}UNEXPECTED FAILURE (was expected to pass){RESET}")
        elif not config.expected_to_pass and result['status'] == 'PASSED':
            print(f"    {GREEN}UNEXPECTED SUCCESS (was expected to fail){RESET}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] in ['FAILED', 'TIMEOUT', 'ERROR'])
    skipped = sum(1 for r in results if r['status'] == 'SKIPPED')
    
    print(f"{GREEN}PASSED: {passed}{RESET}")
    print(f"{RED}FAILED/TIMEOUT: {failed}{RESET}")
    print(f"{YELLOW}SKIPPED: {skipped}{RESET}")
    
    print("\nDetailed Results:")
    print("-" * 80)
    for r in results:
        status = r['status']
        if status == 'PASSED':
            status_color = GREEN
        elif status == 'SKIPPED':
            status_color = YELLOW
        else:
            status_color = RED
        print(f"  {r['name']}: {status_color}{status}{RESET}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
