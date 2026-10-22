# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compare Twinkle vLLMSampler with Swift VllmEngine.

Usage:
    python align_swift.py                 # local LLM comparison
    python align_swift.py --multimodal    # also compare multimodal inference
    python align_swift.py --ray           # Ray placement benchmark (6 GPUs)

The Ray benchmark uses 4 GPUs for a model and 2 for samplers; it measures
sampling with a separate model group, without training or weight sync.
"""

import argparse
import gc
import sys
import torch
from swift.infer_engine import RequestConfig, VllmEngine
from swift.utils import seed_everything

# Do not init twinkle at import so --ray can init with Ray; other tests init local in main.
import twinkle
from twinkle.data_format import SamplingParams, Trajectory
from twinkle.sampler.vllm_sampler import vLLMSampler
from twinkle.template import Qwen3_5Template, Template

# Test models
LLM_MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'
MLLM_MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'

# Test data
LLM_MESSAGES = [{'role': 'user', 'content': '详细地介绍人工智能，越长越好'}]
MLLM_MESSAGES = [{'role': 'user', 'content': '<image>这是什么'}]
MLLM_IMAGES = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']

# vLLM settings for MLLM (to avoid OOM)
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEM = 0.9

SYSTEM_PROMPT = """You are a helpful math assistant. Solve the problem step by step. Show your reasoning in
<think> </think> tags, then give the final numerical answer after ####.
For example:
<think> ... reasoning ... </think>
#### 42"""
GSM8K_MESSAGES1 = [{
    'role': 'system',
    'content': SYSTEM_PROMPT
}, {
    'role':
    'user',
    'content':
    'James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?'
}]
GSM8K_MESSAGES2 = [{
    'role': 'system',
    'content': SYSTEM_PROMPT
}, {
    'role':
    'user',
    'content':
    'Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, '
    'and there are 80% more of those in purple. There are only 25\\% \as many green flowers as there are yellow '
    'and purple flowers. How many flowers does Mark have in his garden?'
}]
GSM8K_MESSAGES3 = [{
    'role': 'system',
    'content': SYSTEM_PROMPT
}, {
    'role':
    'user',
    'content':
    'A car is driving through a tunnel with many turns. After a while, the car must travel through a ring '
    'that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, '
    'it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn,'
    ' it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring,'
    ' how far did it have to travel after the 3rd turn?'
}]
GSM8K_MESSAGES4 = [{
    'role': 'system',
    'content': SYSTEM_PROMPT
}, {
    'role':
    'user',
    'content':
    'Hans booked a room in a hotel. The hotel has 10 floors with 10 identical rooms on each floor. '
    'Because of an accident, the last floor is unavailable for the guests. Considering there are no other guests, '
    'in how many different rooms could Hans be checked in?'
}]


def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()


def test_llm_vllm_sampler():
    seed_everything(42)
    import time
    swift_engine = VllmEngine(LLM_MODEL_ID, gpu_memory_utilization=0.5)
    request_config = RequestConfig(max_tokens=2048, temperature=0, repetition_penalty=1)
    st_time = time.time()
    swift_resp = swift_engine.infer([{'messages': LLM_MESSAGES}] * 16, request_config=request_config)
    swift_response = swift_resp[0].choices[0].message.content
    end_time = time.time()
    print(f'Swift inference time: {end_time - st_time} seconds')
    del swift_engine
    clean_cache()

    seed_everything(42)
    sampler = vLLMSampler(LLM_MODEL_ID, engine_args={'gpu_memory_utilization': 0.5})
    sampler.set_template(Template, model_id=LLM_MODEL_ID)

    trajectory = Trajectory(messages=LLM_MESSAGES)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0, repetition_penalty=1)
    st_time = time.time()
    resp = sampler.sample([trajectory] * 16, sampling_params=sampling_params)
    end_time = time.time()
    print(f'Twinkle inference time: {end_time - st_time} seconds')
    tokens = resp[0].sequences[0].tokens
    twinkle_response = sampler.template.decode(tokens, skip_special_tokens=True)
    del sampler
    clean_cache()

    match = swift_response == twinkle_response
    if not match:
        print(f'Swift: {swift_response}')
        print(f'Twinkle: {twinkle_response}')
    return match


def test_llm_vllm_sampler_ray():
    """Twinkle sampler with Ray + model group (4 GPUs) + sampler group (2 GPUs).

    Isolates RL-like setup (no training/dataset): same 16 requests as local test,
    to measure the impact of Ray and multi-process sampling on sample speed.
    Run alone: python align_swift.py --ray  (requires 6 GPUs).
    """
    import time
    from peft import LoraConfig

    from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
    from twinkle.model import TransformersModel
    from twinkle.processor import InputProcessor

    logger = get_logger()
    MODEL_GPUS = 4
    SAMPLER_GPUS = 2
    NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS
    ADAPTER_NAME = 'default'

    seed_everything(42)

    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU', gpus_per_worker=1),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU', gpus_per_worker=1),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)
    logger.info(get_device_placement())

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    model = TransformersModel(
        model_id=LLM_MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
    )
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME)
    model.set_template('Template', model_id=LLM_MODEL_ID, adapter_name=ADAPTER_NAME)

    sampler = vLLMSampler(
        model_id=LLM_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.5,
            'max_model_len': 4096,
            'max_lora_rank': 64,
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=LLM_MODEL_ID)

    sampler.reset_prefix_cache()

    trajectory = Trajectory(messages=LLM_MESSAGES)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0, repetition_penalty=1)
    trajectories = [trajectory] * 16

    t0 = time.perf_counter()
    sampler.sample(trajectories, sampling_params=sampling_params, adapter_name=ADAPTER_NAME)
    t1 = time.perf_counter()

    print(f'Twinkle Ray (model={MODEL_GPUS}, sampler={SAMPLER_GPUS}) inference time: {t1 - t0:.2f} s')

    # No Swift baseline in same process; compare with local test run separately
    logger.info('Run test_llm_vllm_sampler (local) for baseline comparison.')
    return True


def test_mllm_vllm_sampler():
    seed_everything(42)
    swift_engine = VllmEngine(MLLM_MODEL_ID, gpu_memory_utilization=VLLM_GPU_MEM, max_model_len=VLLM_MAX_MODEL_LEN)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    swift_resp = swift_engine.infer([{'messages': MLLM_MESSAGES, 'images': MLLM_IMAGES}], request_config=request_config)
    swift_response = swift_resp[0].choices[0].message.content
    del swift_engine
    clean_cache()

    seed_everything(42)
    sampler = vLLMSampler(
        MLLM_MODEL_ID, engine_args={'gpu_memory_utilization': VLLM_GPU_MEM, 'max_model_len': VLLM_MAX_MODEL_LEN})
    sampler.set_template(Qwen3_5Template, model_id=MLLM_MODEL_ID)

    trajectory = Trajectory(messages=MLLM_MESSAGES, images=MLLM_IMAGES)
    sampling_params = SamplingParams(max_tokens=128, temperature=0)
    resp = sampler.sample([trajectory], sampling_params=sampling_params)
    tokens = resp[0].sequences[0].tokens
    twinkle_response = sampler.template.decode(tokens, skip_special_tokens=True)
    del sampler
    clean_cache()

    match = swift_response == twinkle_response
    if not match:
        print(f'Swift: {swift_response[:300]}')
        print(f'Twinkle: {twinkle_response[:300]}')
    return match


def main():
    parser = argparse.ArgumentParser(description='Twinkle/Swift vLLM comparisons')
    parser.add_argument('--ray', action='store_true', help='Run only the Ray placement benchmark (6 GPUs)')
    parser.add_argument('--multimodal', action='store_true', help='Include the multimodal comparison')
    args = parser.parse_args()
    if args.ray:
        return 0 if test_llm_vllm_sampler_ray() else 1

    twinkle.initialize(mode='local', nproc_per_node=1)
    results = {'LLM vLLMSampler': test_llm_vllm_sampler()}
    if args.multimodal:
        results['MLLM vLLMSampler'] = test_mllm_vllm_sampler()
    for test_name, passed in results.items():
        print(f"{test_name}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
