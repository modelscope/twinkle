"""
Standalone inference example using Ray + vLLMSampler with LoRA adapter.

This script demonstrates how to:
1. Initialize Twinkle with Ray for distributed inference
2. Create a vLLMSampler with LoRA enabled on dedicated GPUs
3. Load a LoRA adapter from a local checkpoint path
4. Send prompts (Trajectory format) and collect generated responses

Usage:
    # Single GPU inference
    SAMPLER_GPUS=1 python sample.py

    # Multi-GPU inference (tensor parallel)
    SAMPLER_GPUS=2 python sample.py

    # Use a different model / adapter
    MODEL_ID=/path/to/model LORA_PATH=/path/to/adapter SAMPLER_GPUS=1 python sample.py
"""

import os
from typing import List, Dict, Any

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3.5-4B')
LORA_PATH = os.environ.get('LORA_PATH', '/path/to/lora')
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))


def build_prompts() -> List[Dict[str, Any]]:
    """Build a list of Trajectory dicts (messages format) as prompts."""
    prompts = [
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'What is the capital of France?'},
            ]
        },
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Write a short poem about the moon.'},
            ]
        },
        {
            'messages': [
                {'role': 'user', 'content': 'Solve: 2x + 3 = 11. What is x?'},
            ]
        },
    ]
    return prompts


def main():
    # ── 1. Initialize Twinkle with Ray ──────────────────────────────────
    device_groups = [
        DeviceGroup(name='sampler', ranks=list(range(SAMPLER_GPUS)), device_type='GPU', gpus_per_worker=SAMPLER_GPUS),
    ]
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=SAMPLER_GPUS, groups=device_groups)

    # ── 2. Create vLLMSampler with LoRA enabled ────────────────────────
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.7,
            'max_model_len': 4096,
            'enable_lora': True,
            'max_loras': 1,
            'max_lora_rank': 32,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)
    logger.info(get_device_placement())

    # ── 3. Configure sampling parameters ────────────────────────────────
    sampling_params = SamplingParams(
        max_tokens=2018,
        temperature=0.7,
        top_p=0.9,
        num_samples=1,
    )

    # ── 4. Run inference ────────────────────────────────────────────────
    prompts = build_prompts()
    logger.info(f'Sampling {len(prompts)} prompts with model {MODEL_ID} ...')

    responses = sampler.sample(prompts, sampling_params, adapter_path=LORA_PATH)

    # ── 5. Print results ────────────────────────────────────────────────
    for i, response in enumerate(responses):
        for seq in response.sequences:
            text = sampler.template.tokenizer.decode(seq.tokens, skip_special_tokens=True)
            logger.info(f'\n{"="*60}\nPrompt {i}: {prompts[i]["messages"][-1]["content"]}\n{"─"*60}\n{text}\n')

    logger.info('Done.')


if __name__ == '__main__':
    main()
