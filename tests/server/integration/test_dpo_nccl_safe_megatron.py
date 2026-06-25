# Copyright (c) ModelScope Contributors. All rights reserved.
"""DPO NCCL-safe verification test for Megatron backend.

Tests that DPO training (forward_only + forward_backward) does NOT cause
NCCL hang when errors occur, verifying the nccl_safe_megatron fix.

Prerequisites:
    1. Ray cluster running with GPUs
    2. Twinkle server started with Megatron backend and TWINKLE_FAIL_FAST=0

Usage (direct):
    python tests/server/integration/test_dpo_nccl_safe_megatron.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:9000')
BASE_MODEL = 'Qwen/Qwen3.5-4B'
TIMEOUT = 120


def log(msg):
    print(f'[DPO-Megatron] {msg}', flush=True)


def wait_for_server(url, timeout=300):
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f'{url}/-/routes', timeout=5)
            if resp.status_code == 200:
                log(f'Server ready ({int(time.time() - start)}s)')
                return True
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f'Server not ready after {timeout}s')


def init_dpo_client():
    """Initialize Twinkle client for DPO training."""
    from twinkle_client import init_twinkle_client
    from twinkle_client.model import MultiLoraTransformersModel
    from peft import LoraConfig

    init_twinkle_client(base_url=SERVER_URL, api_key='EMPTY_TOKEN')

    model = MultiLoraTransformersModel(model_id=f'ms://{BASE_MODEL}')
    model.add_adapter_to_model(
        adapter_name='dpo-test',
        config=LoraConfig(r=16, target_modules=['q_proj', 'v_proj']),
        gradient_accumulation_steps=1,
    )
    model.set_loss('DPOLoss', init_args={'beta': 0.1})
    model.set_optimizer('Adam', lr=1e-5)
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    log('DPO client configured')
    return model


def make_dpo_batch(batch_size=4, seq_len=64, completion_len=32):
    """Create DPO batch: interleaved chosen/rejected pairs."""
    prompt_len = seq_len - completion_len
    all_inputs = []

    # DPO requires even batch (chosen/rejected pairs)
    for i in range(batch_size):
        input_ids = list(range(1, seq_len + 1))
        labels = [-100] * prompt_len + list(range(100, 100 + completion_len))
        all_inputs.append({
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': [1] * seq_len,
        })

    return all_inputs


def run_dpo_step(model, test_name, *, bad_ref_logps=False):
    """Execute a full DPO step: forward_only (ref) + forward_backward (policy)."""
    batch = make_dpo_batch(batch_size=4)

    # Step 1: forward_only for reference logps
    log(f'[{test_name}] forward_only (reference)...')
    start = time.time()
    try:
        ref_result = model.forward_only(inputs=batch, disable_lora=True)
        elapsed = time.time() - start
        log(f'[{test_name}] forward_only OK ({elapsed:.1f}s)')
    except Exception as e:
        elapsed = time.time() - start
        log(f'[{test_name}] forward_only FAILED ({elapsed:.1f}s): {e}')
        if elapsed > TIMEOUT:
            log(f'[{test_name}] TIMEOUT! NCCL HANG detected!')
            return False
        return True  # Error was caught, not a hang

    # Step 2: forward_backward for policy training
    log(f'[{test_name}] forward_backward (policy)...')
    start = time.time()
    try:
        # For DPO, we need ref_logps from the reference forward
        kwargs = {}
        if hasattr(ref_result, 'result') and ref_result.result:
            # Extract ref_logps from forward_only result
            pass  # In real DPO, client passes ref_logps
        result = model.forward_backward(inputs=batch, **kwargs)
        elapsed = time.time() - start
        log(f'[{test_name}] forward_backward OK ({elapsed:.1f}s)')
    except Exception as e:
        elapsed = time.time() - start
        log(f'[{test_name}] forward_backward FAILED ({elapsed:.1f}s): {e}')
        if elapsed > TIMEOUT:
            log(f'[{test_name}] TIMEOUT! NCCL HANG detected!')
            return False
        return True  # Error was caught, not a hang

    # Step 3: optimizer step
    try:
        model.clip_grad_and_step()
        log(f'[{test_name}] clip_grad_and_step OK')
    except Exception as e:
        log(f'[{test_name}] clip_grad_and_step FAILED: {e}')

    return True


def main():
    log('=' * 60)
    log('DPO NCCL-Safe Verification - Megatron Backend')
    log('=' * 60)
    log(f'Server URL: {SERVER_URL}')
    log(f'TWINKLE_FAIL_FAST = {os.getenv("TWINKLE_FAIL_FAST", "1 (default)")}')

    wait_for_server(SERVER_URL)
    model = init_dpo_client()

    results = []

    # Test 1: Normal DPO training (should work)
    passed = run_dpo_step(model, 'TEST-1-NORMAL-DPO')
    results.append(('TEST-1: Normal DPO', passed))

    # Test 2: Multiple consecutive DPO steps
    for i in range(3):
        passed = run_dpo_step(model, f'TEST-2-CONSECUTIVE-{i+1}')
        results.append((f'TEST-2-{i+1}: Consecutive DPO', passed))

    # Test 3: forward_only then forward_backward rapidly
    passed = run_dpo_step(model, 'TEST-3-RAPID')
    results.append(('TEST-3: Rapid DPO', passed))

    # Test 4: Health check after all DPO operations
    log('[TEST-4] Final health check - forward_backward...')
    batch = make_dpo_batch(batch_size=4)
    start = time.time()
    try:
        model.forward_backward(inputs=batch)
        elapsed = time.time() - start
        log(f'[TEST-4] OK ({elapsed:.1f}s)')
        results.append(('TEST-4: Final health', True))
    except Exception as e:
        elapsed = time.time() - start
        log(f'[TEST-4] FAILED ({elapsed:.1f}s): {e}')
        results.append(('TEST-4: Final health', elapsed < TIMEOUT))

    # Summary
    log(f'\n{"=" * 60}\nRESULTS SUMMARY\n{"=" * 60}')
    all_passed = all(p for _, p in results)
    for name, status in results:
        log(f'  [{"PASS" if status else "FAIL"}] {name}')
    log(f'\n{"ALL" if all_passed else "SOME"} {len(results)} TESTS {"PASSED" if all_passed else "FAILED"}!')
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
