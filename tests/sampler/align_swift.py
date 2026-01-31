# Copyright (c) ModelScope Contributors. All rights reserved.
"""Alignment tests between twinkle samplers and swift inference engines.

This script tests that twinkle's TorchSampler and VLLMSampler produce identical
results to swift's TransformersEngine and VllmEngine respectively.

Test cases:
1. LLM + TorchSampler vs TransformersEngine
2. LLM + VLLMSampler vs VllmEngine
3. MLLM + TorchSampler vs TransformersEngine
4. MLLM + VLLMSampler vs VllmEngine
"""

import gc
import torch

import twinkle
twinkle.initialize(mode='local', nproc_per_node=1)

from swift.infer_engine import TransformersEngine, VllmEngine, RequestConfig
from swift.utils import seed_everything
from twinkle.sampler import TorchSampler, VLLMSampler, SamplingParams
from twinkle.template import Template
from twinkle.template.qwen3_vl import Qwen3VLTemplate
from twinkle.data_format import Trajectory

# Test models
LLM_MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'
MLLM_MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'

# Test data
LLM_MESSAGES = [{'role': 'user', 'content': '你好，请用一句话介绍你自己'}]
MLLM_MESSAGES = [{'role': 'user', 'content': '<image>这是什么'}]
MLLM_IMAGES = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']

# vLLM settings for MLLM (to avoid OOM)
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEM = 0.9


def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()


def test_llm_torch_sampler():

    seed_everything(42)
    swift_engine = TransformersEngine(LLM_MODEL_ID)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    swift_resp = swift_engine.infer([{'messages': LLM_MESSAGES}], request_config=request_config)
    swift_response = swift_resp[0].choices[0].message.content
    del swift_engine
    clean_cache()
    
    # Twinkle inference
    seed_everything(42)
    sampler = TorchSampler(LLM_MODEL_ID)
    sampler.set_template(Template, model_id=LLM_MODEL_ID)
    
    trajectory = Trajectory(messages=LLM_MESSAGES)
    sampling_params = SamplingParams(max_tokens=128, temperature=0)
    resp = sampler.sample([trajectory], sampling_params=sampling_params)
    tokens = resp.sequences[0].tokens
    twinkle_response = sampler.template.decode(tokens, skip_special_tokens=True)
    del sampler
    clean_cache()

    match = swift_response == twinkle_response
    if not match:
        print(f'Swift: {swift_response}')
        print(f'Twinkle: {twinkle_response}')

    return match


def test_llm_vllm_sampler():
    seed_everything(42)
    swift_engine = VllmEngine(LLM_MODEL_ID, gpu_memory_utilization=0.5)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    swift_resp = swift_engine.infer([{'messages': LLM_MESSAGES}], request_config=request_config)
    swift_response = swift_resp[0].choices[0].message.content
    del swift_engine
    clean_cache()

    seed_everything(42)
    sampler = VLLMSampler(LLM_MODEL_ID, gpu_memory_utilization=0.5)
    sampler.set_template(Template, model_id=LLM_MODEL_ID)
    
    trajectory = Trajectory(messages=LLM_MESSAGES)
    sampling_params = SamplingParams(max_tokens=128, temperature=0)
    resp = sampler.sample([trajectory], sampling_params=sampling_params)
    tokens = resp.sequences[0].tokens
    twinkle_response = sampler.template.decode(tokens, skip_special_tokens=True)
    del sampler
    clean_cache()
    
    match = swift_response == twinkle_response
    if not match:
        print(f'Swift: {swift_response}')
        print(f'Twinkle: {twinkle_response}')
    return match


def test_mllm_torch_sampler():
    seed_everything(42)
    swift_engine = TransformersEngine(MLLM_MODEL_ID)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    swift_resp = swift_engine.infer([{'messages': MLLM_MESSAGES, 'images': MLLM_IMAGES}], request_config=request_config)
    swift_response = swift_resp[0].choices[0].message.content
    del swift_engine
    clean_cache()

    seed_everything(42)
    from transformers import Qwen3VLForConditionalGeneration
    sampler = TorchSampler(MLLM_MODEL_ID, model_cls=Qwen3VLForConditionalGeneration)
    sampler.set_template(Qwen3VLTemplate, model_id=MLLM_MODEL_ID)
    
    trajectory = Trajectory(messages=MLLM_MESSAGES, images=MLLM_IMAGES)
    sampling_params = SamplingParams(max_tokens=128, temperature=0)
    resp = sampler.sample([trajectory], sampling_params=sampling_params)
    tokens = resp.sequences[0].tokens
    twinkle_response = sampler.template.decode(tokens, skip_special_tokens=True)
    del sampler
    clean_cache()
    
    match = swift_response == twinkle_response
    if not match:
        print(f'Swift: {swift_response[:300]}')
        print(f'Twinkle: {twinkle_response[:300]}')
    return match


def test_mllm_vllm_sampler():
    seed_everything(42)
    swift_engine = VllmEngine(MLLM_MODEL_ID, gpu_memory_utilization=VLLM_GPU_MEM, max_model_len=VLLM_MAX_MODEL_LEN)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    swift_resp = swift_engine.infer([{'messages': MLLM_MESSAGES, 'images': MLLM_IMAGES}], request_config=request_config)
    swift_response = swift_resp[0].choices[0].message.content
    del swift_engine
    clean_cache()

    seed_everything(42)
    sampler = VLLMSampler(MLLM_MODEL_ID, gpu_memory_utilization=VLLM_GPU_MEM, max_model_len=VLLM_MAX_MODEL_LEN)
    sampler.set_template(Qwen3VLTemplate, model_id=MLLM_MODEL_ID)
    
    trajectory = Trajectory(messages=MLLM_MESSAGES, images=MLLM_IMAGES)
    sampling_params = SamplingParams(max_tokens=128, temperature=0)
    resp = sampler.sample([trajectory], sampling_params=sampling_params)
    tokens = resp.sequences[0].tokens
    twinkle_response = sampler.template.decode(tokens, skip_special_tokens=True)
    del sampler
    clean_cache()
    
    match = swift_response == twinkle_response
    if not match:
        print(f'Swift: {swift_response[:300]}')
        print(f'Twinkle: {twinkle_response[:300]}')
    return match


def main():
    results = {}
    
    # Run all tests
    results['LLM TorchSampler'] = test_llm_torch_sampler()
    results['LLM VLLMSampler'] = test_llm_vllm_sampler()
    results['MLLM TorchSampler'] = test_mllm_torch_sampler()
    results['MLLM VLLMSampler'] = test_mllm_vllm_sampler()

    for test_name, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'{test_name}: {status}')
    
    all_passed = all(results.values())
    print(f'\nAll tests passed: {all_passed}')
    return all_passed


if __name__ == '__main__':
    main()
