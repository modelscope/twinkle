# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real E2E test for NCCL-safe fault tolerance via Twinkle client path.

Exercises the /twinkle/forward_backward endpoint through the Twinkle SDK
(init_twinkle_client + MultiLoraTransformersModel). This is a SEPARATE code
path from the Tinker SDK (/tinker/forward_backward).

Prerequisites:
    1. Ray cluster running with GPUs (2 for model DP/TP, optionally 1 for sampler)
    2. Twinkle server started with TWINKLE_FAIL_FAST=0

Usage (direct):
    python tests/server/integration/test_nccl_safe_twinkle_e2e.py

Usage (pytest, requires TWINKLE_TEST_GPU_E2E=1):
    TWINKLE_TEST_GPU_E2E=1 pytest tests/server/integration/test_nccl_safe_twinkle_e2e.py -v
"""
from __future__ import annotations

import os
import sys
import time
import logging
import traceback
from typing import Any, Dict, List

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Set TWINKLE_TEST_GPU_E2E=1 to run real GPU E2E tests (requires running server)',
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def log(msg):
    print(f'[E2E-Twinkle] {msg}', flush=True)


BASE_MODEL = 'Qwen/Qwen3.5-4B'
SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:9000')
TIMEOUT = 120
ADAPTER_NAME = 'nccl-safe-test'


def wait_for_server(url, timeout=300):
    """Wait for Twinkle server to become ready."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f'{url}/-/routes', timeout=5)
            if resp.status_code == 200:
                log(f'Server is ready (waited {int(time.time() - start)}s)')
                return True
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f'Server not ready after {timeout}s')


def init_client():
    """Initialize Twinkle client and configure model for GRPO training."""
    from twinkle_client import init_twinkle_client
    from twinkle_client.model import MultiLoraTransformersModel
    from peft import LoraConfig

    init_twinkle_client(base_url=SERVER_URL, api_key='EMPTY_TOKEN')

    model = MultiLoraTransformersModel(model_id=f'ms://{BASE_MODEL}')
    model.add_adapter_to_model(
        adapter_name=ADAPTER_NAME,
        config=LoraConfig(r=16, target_modules=['q_proj', 'v_proj']),
        gradient_accumulation_steps=1,
    )
    model.set_loss('GRPOLoss', init_args={'epsilon': 0.2})
    model.set_optimizer('Adam', lr=1e-5)
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    log('Twinkle client + model configured successfully')
    return model


def make_input_features(
    batch_size=4, seq_len=64, completion_len=32, *,
    bad_old_logps_len=None, include_advantages=True,
    nan_old_logps=False, extreme_advantages=None, all_labels_masked=False,
):
    """Construct InputFeature list + old_logps + advantages for GRPO."""
    prompt_len = seq_len - completion_len
    input_features = []
    old_logps_list = []
    advantages_list = []

    for i in range(batch_size):
        input_ids = list(range(1, seq_len + 1))
        labels = [-100] * seq_len if all_labels_masked else (
            [-100] * prompt_len + list(range(100, 100 + completion_len)))
        input_features.append({
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': [1] * seq_len,
            'position_ids': list(range(seq_len)),
        })

        if bad_old_logps_len is not None:
            logps = np.random.randn(bad_old_logps_len).tolist()
        elif nan_old_logps:
            logps = [float('nan')] * completion_len
        else:
            logps = np.random.randn(completion_len).tolist()
        old_logps_list.append(logps)

        if extreme_advantages is not None:
            advantages_list.append(extreme_advantages if i % 2 == 0 else -extreme_advantages)
        else:
            advantages_list.append(float(np.random.randn()))

    old_logps = old_logps_list if include_advantages else None
    advantages = advantages_list if include_advantages else None
    return input_features, old_logps, advantages


def run_forward_backward(model, inputs, old_logps, advantages, test_name):
    """Run forward_backward and return (success, result, elapsed_seconds)."""
    log(f'[{test_name}] Sending {len(inputs)} input features...')
    start = time.time()
    try:
        kwargs: Dict[str, Any] = {}
        if old_logps is not None:
            kwargs['old_logps'] = old_logps
        if advantages is not None:
            kwargs['advantages'] = advantages

        result = model.forward_backward(inputs=inputs, **kwargs)
        elapsed = time.time() - start
        log(f'[{test_name}] Completed in {elapsed:.1f}s')
        if hasattr(result, 'result') and result.result is not None:
            log(f'[{test_name}] result = {result.result}')
        return True, result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        log(f'[{test_name}] FAILED in {elapsed:.1f}s: {type(e).__name__}: {e}')
        if elapsed > TIMEOUT:
            log(f'[{test_name}] TIMEOUT! This suggests NCCL hang!')
        return False, None, elapsed


def do_optim_step(model, test_name):
    """Run clip_grad_and_step."""
    try:
        model.clip_grad_and_step()
        log(f'[{test_name}] clip_grad_and_step OK')
        return True
    except Exception as e:
        log(f'[{test_name}] clip_grad_and_step FAILED: {e}')
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Test Scenarios (12 tests)
# ═══════════════════════════════════════════════════════════════════════════

def test_1_normal_grpo(m):
    inputs, old_logps, adv = make_input_features(batch_size=4)
    ok, _, elapsed = run_forward_backward(m, inputs, old_logps, adv, 'TEST-1-NORMAL')
    assert ok and elapsed < TIMEOUT
    do_optim_step(m, 'TEST-1-NORMAL')
    return True

def test_2_bad_old_logps(m):
    inputs, old_logps, adv = make_input_features(batch_size=4, bad_old_logps_len=5)
    ok, _, elapsed = run_forward_backward(m, inputs, old_logps, adv, 'TEST-2-BAD-LOGPS')
    if not ok:
        return elapsed < TIMEOUT
    assert elapsed < TIMEOUT
    do_optim_step(m, 'TEST-2-BAD-LOGPS')
    return True

def test_3_recovery(m):
    inputs, old_logps, adv = make_input_features(batch_size=4)
    ok, _, elapsed = run_forward_backward(m, inputs, old_logps, adv, 'TEST-3-RECOVERY')
    assert ok and elapsed < TIMEOUT
    do_optim_step(m, 'TEST-3-RECOVERY')
    return True

def test_4_nan_old_logps(m):
    inputs, old_logps, adv = make_input_features(batch_size=4, nan_old_logps=True)
    _, _, elapsed = run_forward_backward(m, inputs, old_logps, adv, 'TEST-4-NAN')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(m, 'TEST-4-NAN')
    return True

def test_5_extreme_advantages(m):
    inputs, old_logps, adv = make_input_features(batch_size=4, extreme_advantages=1e30)
    _, _, elapsed = run_forward_backward(m, inputs, old_logps, adv, 'TEST-5-EXTREME')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(m, 'TEST-5-EXTREME')
    return True

def test_6_all_labels_masked(m):
    inputs, old_logps, adv = make_input_features(batch_size=4, all_labels_masked=True)
    _, _, elapsed = run_forward_backward(m, inputs, old_logps, adv, 'TEST-6-MASKED')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(m, 'TEST-6-MASKED')
    return True

def test_7_consecutive_bad(m):
    for i in range(5):
        inputs, old_logps, adv = make_input_features(batch_size=4, bad_old_logps_len=i+1)
        _, _, elapsed = run_forward_backward(m, inputs, old_logps, adv, f'TEST-7-{i+1}')
        if elapsed >= TIMEOUT:
            return False
        do_optim_step(m, f'TEST-7-{i+1}')
    return True

def test_8_rapid_bad_good(m):
    for i in range(5):
        bad_in, bad_lp, bad_adv = make_input_features(batch_size=4, bad_old_logps_len=i+1)
        _, _, elapsed = run_forward_backward(m, bad_in, bad_lp, bad_adv, f'TEST-8-BAD-{i+1}')
        if elapsed >= TIMEOUT:
            return False
        do_optim_step(m, f'TEST-8-BAD-{i+1}')
        good_in, good_lp, good_adv = make_input_features(batch_size=4)
        ok, _, elapsed = run_forward_backward(m, good_in, good_lp, good_adv, f'TEST-8-GOOD-{i+1}')
        if not ok or elapsed >= TIMEOUT:
            return False
        do_optim_step(m, f'TEST-8-GOOD-{i+1}')
    return True

def test_9_final_health(m):
    inputs, old_logps, adv = make_input_features(batch_size=4)
    ok, _, elapsed = run_forward_backward(m, inputs, old_logps, adv, 'TEST-9-FINAL')
    assert ok and elapsed < TIMEOUT
    do_optim_step(m, 'TEST-9-FINAL')
    return True

def test_10_gradient_accumulation_error(m):
    inputs, lp, adv = make_input_features(batch_size=4)
    ok, _, elapsed = run_forward_backward(m, inputs, lp, adv, 'TEST-10-GA1')
    if not ok or elapsed >= TIMEOUT:
        return False
    bad_in, bad_lp, bad_adv = make_input_features(batch_size=4, bad_old_logps_len=3)
    _, _, elapsed = run_forward_backward(m, bad_in, bad_lp, bad_adv, 'TEST-10-GA2-BAD')
    if elapsed >= TIMEOUT:
        return False
    inputs, lp, adv = make_input_features(batch_size=4)
    ok, _, elapsed = run_forward_backward(m, inputs, lp, adv, 'TEST-10-GA3')
    if not ok or elapsed >= TIMEOUT:
        return False
    do_optim_step(m, 'TEST-10-GA')
    return True

def test_11_forward_only_then_train(m):
    inputs, _, _ = make_input_features(batch_size=4, include_advantages=False)
    start = time.time()
    try:
        m.forward_only(inputs=inputs)
    except Exception:
        if time.time() - start >= TIMEOUT:
            return False
    train_in, lp, adv = make_input_features(batch_size=4)
    ok, _, elapsed = run_forward_backward(m, train_in, lp, adv, 'TEST-11-TRAIN')
    if not ok or elapsed >= TIMEOUT:
        return False
    do_optim_step(m, 'TEST-11-TRAIN')
    return True

def test_12_mixed_seq_lengths(m):
    all_inputs, all_lp, all_adv = [], [], []
    for sl, cl in [(32, 16), (128, 64), (48, 24), (96, 48)]:
        feats, lp, adv = make_input_features(batch_size=1, seq_len=sl, completion_len=cl)
        all_inputs.extend(feats)
        if lp:
            all_lp.extend(lp)
        if adv:
            all_adv.extend(adv)
    _, _, elapsed = run_forward_backward(m, all_inputs, all_lp, all_adv, 'TEST-12-MIXED')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(m, 'TEST-12-MIXED')
    return True


ALL_TESTS = [
    ('TEST-1: Normal GRPO Training', test_1_normal_grpo),
    ('TEST-2: Bad old_logps (original bug)', test_2_bad_old_logps),
    ('TEST-3: Recovery after error', test_3_recovery),
    ('TEST-4: NaN old_logps', test_4_nan_old_logps),
    ('TEST-5: Extreme advantages (1e30)', test_5_extreme_advantages),
    ('TEST-6: All labels masked (-100)', test_6_all_labels_masked),
    ('TEST-7: Consecutive bad batches', test_7_consecutive_bad),
    ('TEST-8: Rapid bad->good', test_8_rapid_bad_good),
    ('TEST-9: Final health check', test_9_final_health),
    ('TEST-10: Gradient accumulation error', test_10_gradient_accumulation_error),
    ('TEST-11: forward_only then train', test_11_forward_only_then_train),
    ('TEST-12: Mixed sequence lengths', test_12_mixed_seq_lengths),
]


def main():
    log('=' * 60)
    log('NCCL-Safe E2E Test - Twinkle Client Path')
    log('=' * 60)
    log(f'Server URL: {SERVER_URL}')
    log(f'Base Model: {BASE_MODEL}')
    log(f'TWINKLE_FAIL_FAST = {os.getenv("TWINKLE_FAIL_FAST", "1 (default)")}')

    wait_for_server(SERVER_URL)
    m = init_client()

    results = []
    for name, test_fn in ALL_TESTS:
        log(f'\n{"=" * 60}\n{name}\n{"=" * 60}')
        try:
            passed = test_fn(m)
            results.append((name, 'PASS' if passed else 'FAIL'))
            log(f'[{name}] {"PASS" if passed else "FAIL"}')
        except Exception as e:
            log(f'{name}: EXCEPTION: {e}')
            traceback.print_exc()
            results.append((name, 'FAIL'))

    log(f'\n{"=" * 60}\nRESULTS SUMMARY\n{"=" * 60}')
    all_passed = all(s == 'PASS' for _, s in results)
    for name, status in results:
        log(f'  [{status}] {name}')
    log(f'\n{"ALL" if all_passed else "SOME"} {len(results)} TESTS {"PASSED" if all_passed else "FAILED"}!')
    return 0 if all_passed else 1


def test_nccl_safe_twinkle_e2e():
    """Pytest-collected entry point."""
    rc = main()
    assert rc == 0, 'Some Twinkle NCCL-safe E2E tests failed'


if __name__ == '__main__':
    sys.exit(main())
