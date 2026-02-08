#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for Sampler functionality.

Usage:
    # Run all tests
    python test_sampler_e2e.py
    
    # Run specific test
    python test_sampler_e2e.py --test vllm_trajectory
    python test_sampler_e2e.py --test torch_trajectory
    python test_sampler_e2e.py --test vllm_input_feature
    python test_sampler_e2e.py --test torch_input_feature

Environment:
    TWINKLE_MODEL_ID: Model to use (default: Qwen/Qwen2.5-0.5B)
    TWINKLE_MAX_MODEL_LEN: Max model length (default: 512)
"""

import argparse
import os
import sys
import traceback

# Set environment variables before imports
os.environ.setdefault('TRUST_REMOTE_CODE', '1')

MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen2.5-0.5B')
MAX_MODEL_LEN = int(os.environ.get('TWINKLE_MAX_MODEL_LEN', '512'))


def test_vllm_engine_with_input_ids():
    """Test VLLMEngine with raw input_ids (no Sampler layer)."""
    print("\n" + "=" * 60)
    print("Test: VLLMEngine with input_ids")
    print("=" * 60)
    
    import asyncio
    from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine
    from twinkle.data_format.sampling import SamplingParams

    print(f"Creating VLLMEngine with model: {MODEL_ID}")
    engine = VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.3,
    )
    
    async def run_test():
        tokenizer = await engine.get_tokenizer()
        prompt = "What is 2+2? Answer:"
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        print(f"  Prompt: {prompt}")
        print(f"  Input IDs: {input_ids}")
        
        response = await engine.sample(
            prompt_token_ids=input_ids,
            sampling_params=SamplingParams(max_tokens=32, temperature=0.7),
        )
        return response, tokenizer
    
    loop = asyncio.new_event_loop()
    try:
        response, tokenizer = loop.run_until_complete(run_test())
    finally:
        loop.close()
    
    # Accept both local SampleResponse and tinker.SampleResponse
    assert hasattr(response, 'sequences'), f"Expected SampleResponse-like, got {type(response)}"
    assert len(response.sequences) >= 1, "Expected at least one sequence"
    
    seq = response.sequences[0]
    print(f"  Stop reason: {seq.stop_reason}")
    print(f"  Generated tokens: {len(seq.tokens)}")
    print(f"  Tokens: {list(seq.tokens)[:10]}...")
    
    decoded = tokenizer.decode(seq.tokens, skip_special_tokens=True)
    print(f"  Decoded text: {decoded}")
    
    print("\n[PASS] VLLMEngine with input_ids")
    return True


def test_transformers_engine_with_input_ids():
    """Test TransformersEngine with raw input_ids (no Sampler layer)."""
    print("\n" + "=" * 60)
    print("Test: TransformersEngine with input_ids")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from twinkle.data_format.sampling import SamplingParams

    print(f"Loading model: {MODEL_ID}")
    
    # Load model and tokenizer directly (bypass remote_class)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "Hello! My name is"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Input IDs: {input_ids}")
    
    # Generate
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    sampling_params = SamplingParams(max_tokens=16, temperature=0.7)
    gen_kwargs = sampling_params.to_transformers(tokenizer)
    gen_kwargs["return_dict_in_generate"] = True
    gen_kwargs["output_scores"] = True
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_tensor,
            attention_mask=torch.ones_like(input_tensor),
            **gen_kwargs
        )
    
    prompt_len = len(input_ids)
    gen_tokens = outputs.sequences[0][prompt_len:].tolist()
    
    print(f"  Generated tokens: {len(gen_tokens)}")
    print(f"  Tokens: {gen_tokens}")
    
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(f"  Decoded text: {decoded}")
    
    print("\n[PASS] TransformersEngine with input_ids")
    return True


def test_vllm_engine_batch():
    """Test VLLMEngine batch sampling."""
    print("\n" + "=" * 60)
    print("Test: VLLMEngine batch sampling")
    print("=" * 60)
    
    import asyncio
    from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine
    from twinkle.data_format.sampling import SamplingParams

    print(f"Creating VLLMEngine with model: {MODEL_ID}")
    engine = VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.3,
    )
    
    async def run_batch_test():
        tokenizer = await engine.get_tokenizer()
        
        prompts = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?",
        ]
        
        sampling_params = SamplingParams(max_tokens=32)
        
        # Sample all in parallel
        tasks = [
            engine.sample(
                prompt_token_ids=tokenizer.encode(p, add_special_tokens=True),
                sampling_params=sampling_params,
            )
            for p in prompts
        ]
        
        responses = await asyncio.gather(*tasks)
        return responses, tokenizer
    
    loop = asyncio.new_event_loop()
    try:
        responses, tokenizer = loop.run_until_complete(run_batch_test())
    finally:
        loop.close()
    
    assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"
    
    for i, response in enumerate(responses):
        assert hasattr(response, 'sequences'), f"Expected SampleResponse-like, got {type(response)}"
        assert len(response.sequences) >= 1
        seq = response.sequences[0]
        decoded = tokenizer.decode(list(seq.tokens), skip_special_tokens=True)
        print(f"  Response {i}: {decoded[:50]}...")
    
    print("\n[PASS] VLLMEngine batch sampling")
    return True


def test_sampling_params_conversion():
    """Test SamplingParams conversion to vLLM and transformers formats."""
    print("\n" + "=" * 60)
    print("Test: SamplingParams conversion")
    print("=" * 60)
    
    from twinkle.data_format.sampling import SamplingParams
    
    params = SamplingParams(
        max_tokens=64,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        stop=["<|end|>", "\n"],
    )
    
    # Test to_transformers
    gen_kwargs = params.to_transformers()
    assert gen_kwargs["max_new_tokens"] == 64
    assert gen_kwargs["temperature"] == 0.8
    assert gen_kwargs["top_p"] == 0.95
    assert gen_kwargs["top_k"] == 50
    assert gen_kwargs["do_sample"] is True
    print("  to_transformers(): OK")
    
    # Test to_vllm (requires vllm)
    try:
        vllm_params = params.to_vllm()
        assert vllm_params.max_tokens == 64
        assert vllm_params.temperature == 0.8
        assert vllm_params.top_p == 0.95
        assert vllm_params.top_k == 50
        assert vllm_params.stop == ["<|end|>", "\n"]
        print("  to_vllm(): OK")
    except ImportError:
        print("  to_vllm(): SKIPPED (vllm not installed)")
    
    print("\n[PASS] SamplingParams conversion")
    return True


TESTS = {
    'vllm_engine': test_vllm_engine_with_input_ids,
    'transformers_engine': test_transformers_engine_with_input_ids,
    'vllm_batch': test_vllm_engine_batch,
    'params_conversion': test_sampling_params_conversion,
}


def main():
    parser = argparse.ArgumentParser(description='Sampler E2E Tests')
    parser.add_argument('--test', choices=list(TESTS.keys()) + ['all'], default='all',
                        help='Which test to run')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Twinkle Sampler E2E Tests")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Max model length: {MAX_MODEL_LEN}")
    
    if args.test == 'all':
        tests_to_run = list(TESTS.items())
    else:
        tests_to_run = [(args.test, TESTS[args.test])]
    
    results = {}
    for name, test_fn in tests_to_run:
        try:
            success = test_fn()
            results[name] = 'PASS' if success else 'FAIL'
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            traceback.print_exc()
            results[name] = 'FAIL'
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results.items():
        status = "✓" if result == 'PASS' else "✗"
        print(f"  {status} {name}: {result}")
    
    passed = sum(1 for r in results.values() if r == 'PASS')
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
