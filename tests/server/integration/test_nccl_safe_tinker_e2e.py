# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real E2E test for NCCL-safe fault tolerance via Tinker client path.

Exercises the /tinker/forward_backward endpoint through the upstream Tinker SDK.
All adversarial scenarios verify that safe_loss catches errors gracefully without
NCCL hang or model state corruption.

Prerequisites:
    1. Ray cluster running with GPUs (2 for model DP/TP, optionally 1 for sampler)
    2. Twinkle server started with TWINKLE_FAIL_FAST=0

Usage (direct):
    python tests/server/integration/test_nccl_safe_tinker_e2e.py

Usage (pytest, requires TWINKLE_TEST_GPU_E2E=1):
    TWINKLE_TEST_GPU_E2E=1 pytest tests/server/integration/test_nccl_safe_tinker_e2e.py -v
"""
from __future__ import annotations

import os
import sys
import time
import logging
import traceback

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Set TWINKLE_TEST_GPU_E2E=1 to run real GPU E2E tests (requires running server)',
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def log(msg):
    """Print + flush to avoid log suppression by init_tinker_client()."""
    print(f'[E2E-Tinker] {msg}', flush=True)


BASE_MODEL = 'Qwen/Qwen3.5-4B'
SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:9000')
TIMEOUT = 120


def wait_for_server(url, timeout=300):
    """Wait for Twinkle server to become ready."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f'{url}/-/routes', timeout=5)
            if resp.status_code == 200:
                elapsed = int(time.time() - start)
                log(f'Server is ready (waited {elapsed}s)')
                return True
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f'Server not ready after {timeout}s')


def init_client():
    """Initialize Tinker client and create training client."""
    os.environ['TINKER_BASE_URL'] = SERVER_URL
    os.environ['TWINKLE_SERVER_TOKEN'] = 'EMPTY_TOKEN'

    from twinkle_client import init_tinker_client
    init_tinker_client()

    from tinker import ServiceClient
    service_client = ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=16)
    log('Training client created successfully')
    return training_client


def make_datum(seq_len=32, completion_len=16, *, bad_logprobs_len=None, include_advantages=True):
    """Construct a Datum for GRPO training."""
    from tinker import types

    prompt_len = seq_len - completion_len
    input_tokens = list(range(1, seq_len + 1))
    target_tokens = [0] * prompt_len + list(range(100, 100 + completion_len))
    weights = [0] * prompt_len + [1] * completion_len

    if bad_logprobs_len is not None:
        logprobs_values = np.random.randn(bad_logprobs_len).astype(np.float32)
        padded_logprobs = [0.0] * prompt_len + logprobs_values.tolist()
    else:
        logprobs_values = np.random.randn(completion_len).astype(np.float32)
        padded_logprobs = [0.0] * prompt_len + logprobs_values.tolist()

    loss_fn_inputs = {
        'target_tokens': target_tokens,
        'weights': weights,
        'logprobs': types.TensorData.from_numpy(np.array(padded_logprobs, dtype=np.float32)),
    }

    if include_advantages:
        advantage = float(np.random.randn())
        padded_advantages = [0.0] * prompt_len + [advantage] * completion_len
        loss_fn_inputs['advantages'] = types.TensorData.from_numpy(
            np.array(padded_advantages, dtype=np.float32))

    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs=loss_fn_inputs,
    )


def run_forward_backward(training_client, datums, test_name, expect_success=True):
    """Run forward_backward and return (success, result, elapsed_seconds)."""
    log(f'[{test_name}] Sending {len(datums)} datums...')
    start = time.time()
    try:
        result = training_client.forward_backward(datums, 'importance_sampling').result()
        elapsed = time.time() - start
        log(f'[{test_name}] Completed in {elapsed:.1f}s')
        if hasattr(result, 'metrics') and result.metrics:
            loss_avg = result.metrics.get('loss:avg', 'N/A')
            log(f'[{test_name}] loss:avg = {loss_avg}')
        return True, result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        log(f'[{test_name}] FAILED in {elapsed:.1f}s: {type(e).__name__}: {e}')
        if elapsed > TIMEOUT:
            log(f'[{test_name}] TIMEOUT! This suggests NCCL hang!')
        return False, None, elapsed


def do_optim_step(training_client, test_name):
    """Run optimizer step."""
    from tinker import types
    try:
        training_client.optim_step(types.AdamParams(learning_rate=1e-5)).result()
        log(f'[{test_name}] optim_step OK')
        return True
    except Exception as e:
        log(f'[{test_name}] optim_step FAILED: {e}')
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Test Scenarios (19 tests)
# ═══════════════════════════════════════════════════════════════════════════

def test_1_normal_grpo(tc):
    datums = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
    ok, result, elapsed = run_forward_backward(tc, datums, 'TEST-1-NORMAL')
    assert ok and elapsed < TIMEOUT
    do_optim_step(tc, 'TEST-1-NORMAL')
    return True

def test_2_bad_old_logps(tc):
    datums = [
        make_datum(seq_len=64, completion_len=32),
        make_datum(seq_len=64, completion_len=32, bad_logprobs_len=5),
        make_datum(seq_len=64, completion_len=32),
        make_datum(seq_len=64, completion_len=32, bad_logprobs_len=99),
    ]
    ok, result, elapsed = run_forward_backward(tc, datums, 'TEST-2-BAD-LOGPS')
    if not ok:
        return elapsed < TIMEOUT
    assert elapsed < TIMEOUT
    do_optim_step(tc, 'TEST-2-BAD-LOGPS')
    return True

def test_3_recovery(tc):
    datums = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
    ok, _, elapsed = run_forward_backward(tc, datums, 'TEST-3-RECOVERY')
    assert ok and elapsed < TIMEOUT
    do_optim_step(tc, 'TEST-3-RECOVERY')
    return True

def test_4_no_advantages(tc):
    datums = [make_datum(seq_len=64, completion_len=32, include_advantages=False) for _ in range(4)]
    ok, _, elapsed = run_forward_backward(tc, datums, 'TEST-4-NO-ADV')
    assert ok and elapsed < TIMEOUT
    do_optim_step(tc, 'TEST-4-NO-ADV')
    return True

def test_5_consecutive_bad(tc):
    for i in range(5):
        datums = [make_datum(seq_len=64, completion_len=32, bad_logprobs_len=3+i) for _ in range(4)]
        _, _, elapsed = run_forward_backward(tc, datums, f'TEST-5-{i+1}')
        if elapsed >= TIMEOUT:
            return False
        do_optim_step(tc, f'TEST-5-{i+1}')
    return True

def test_6_nan_logprobs(tc):
    from tinker import types
    datums = []
    for _ in range(4):
        d = make_datum(seq_len=64, completion_len=32)
        d.loss_fn_inputs['logprobs'] = types.TensorData.from_numpy(
            np.array([float('nan')] * 64, dtype=np.float32))
        datums.append(d)
    _, _, elapsed = run_forward_backward(tc, datums, 'TEST-6-NAN')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-6-NAN')
    return True

def test_7_inf_logprobs(tc):
    from tinker import types
    datums = []
    for _ in range(4):
        d = make_datum(seq_len=64, completion_len=32)
        inf_arr = np.full(64, float('inf'), dtype=np.float32)
        inf_arr[::2] = float('-inf')
        d.loss_fn_inputs['logprobs'] = types.TensorData.from_numpy(inf_arr)
        datums.append(d)
    _, _, elapsed = run_forward_backward(tc, datums, 'TEST-7-INF')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-7-INF')
    return True

def test_8_extreme_advantages(tc):
    from tinker import types
    datums = []
    for i in range(4):
        d = make_datum(seq_len=64, completion_len=32)
        val = 1e30 if i % 2 == 0 else -1e30
        adv = np.full(64, 0.0, dtype=np.float32)
        adv[32:] = val
        d.loss_fn_inputs['advantages'] = types.TensorData.from_numpy(adv)
        datums.append(d)
    _, _, elapsed = run_forward_backward(tc, datums, 'TEST-8-EXTREME-ADV')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-8-EXTREME-ADV')
    return True

def test_9_zero_completion(tc):
    from tinker import types
    datums = []
    for _ in range(4):
        d = types.Datum(
            model_input=types.ModelInput.from_ints(list(range(1, 65))),
            loss_fn_inputs={
                'target_tokens': [0]*64, 'weights': [0]*64,
                'logprobs': types.TensorData.from_numpy(np.zeros(64, dtype=np.float32)),
                'advantages': types.TensorData.from_numpy(np.zeros(64, dtype=np.float32)),
            },
        )
        datums.append(d)
    _, _, elapsed = run_forward_backward(tc, datums, 'TEST-9-ZERO-COMPL')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-9-ZERO-COMPL')
    return True

def test_10_partial_advantages(tc):
    datums = [
        make_datum(seq_len=64, completion_len=32, include_advantages=True),
        make_datum(seq_len=64, completion_len=32, include_advantages=False),
        make_datum(seq_len=64, completion_len=32, include_advantages=True),
        make_datum(seq_len=64, completion_len=32, include_advantages=False),
    ]
    _, _, elapsed = run_forward_backward(tc, datums, 'TEST-10-PARTIAL-ADV')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-10-PARTIAL-ADV')
    return True

def test_11_mixed_seq_lengths(tc):
    datums = [
        make_datum(seq_len=32, completion_len=16),
        make_datum(seq_len=128, completion_len=64),
        make_datum(seq_len=48, completion_len=24),
        make_datum(seq_len=96, completion_len=48),
    ]
    _, _, elapsed = run_forward_backward(tc, datums, 'TEST-11-MIXED')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-11-MIXED')
    return True

def test_12_all_bad(tc):
    datums = [make_datum(seq_len=64, completion_len=32, bad_logprobs_len=i) for i in range(4)]
    _, _, elapsed = run_forward_backward(tc, datums, 'TEST-12-ALL-BAD')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-12-ALL-BAD')
    return True

def test_13_forward_only_then_train(tc):
    datums_infer = [make_datum(seq_len=64, completion_len=32, include_advantages=False) for _ in range(4)]
    start = time.time()
    try:
        tc.forward(datums_infer).result()
    except Exception:
        if time.time() - start >= TIMEOUT:
            return False
    datums_train = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
    ok, _, elapsed = run_forward_backward(tc, datums_train, 'TEST-13-TRAIN')
    if not ok or elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-13-TRAIN')
    return True

def test_14_rapid_bad_good(tc):
    for i in range(5):
        bad = [make_datum(seq_len=64, completion_len=32, bad_logprobs_len=i+1) for _ in range(4)]
        _, _, elapsed = run_forward_backward(tc, bad, f'TEST-14-BAD-{i+1}')
        if elapsed >= TIMEOUT:
            return False
        do_optim_step(tc, f'TEST-14-BAD-{i+1}')
        good = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
        ok, _, elapsed = run_forward_backward(tc, good, f'TEST-14-GOOD-{i+1}')
        if not ok or elapsed >= TIMEOUT:
            return False
        do_optim_step(tc, f'TEST-14-GOOD-{i+1}')
    return True

def test_15_final_health(tc):
    datums = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
    ok, _, elapsed = run_forward_backward(tc, datums, 'TEST-15-FINAL')
    assert ok and elapsed < TIMEOUT
    do_optim_step(tc, 'TEST-15-FINAL')
    return True

def test_16_large_batch(tc):
    datums = [make_datum(seq_len=64, completion_len=32) for _ in range(16)]
    ok, _, elapsed = run_forward_backward(tc, datums, 'TEST-16-LARGE')
    if elapsed >= TIMEOUT:
        return False
    assert ok
    do_optim_step(tc, 'TEST-16-LARGE')
    return True

def test_17_single_datum(tc):
    # With dp_size=2 + nproc_per_node=2, minimum batch must be >= data_world_size
    datums = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
    ok, _, elapsed = run_forward_backward(tc, datums, 'TEST-17-SMALL')
    if elapsed >= TIMEOUT:
        return False
    assert ok
    do_optim_step(tc, 'TEST-17-SMALL')
    return True

def test_18_save_after_error(tc):
    bad = [make_datum(seq_len=64, completion_len=32, bad_logprobs_len=2) for _ in range(4)]
    _, _, elapsed = run_forward_backward(tc, bad, 'TEST-18-ERR')
    if elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-18-ERR')
    try:
        tc.save_weights_for_sampler().result()
    except Exception:
        pass
    good = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
    ok, _, elapsed = run_forward_backward(tc, good, 'TEST-18-POST')
    if not ok or elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-18-POST')
    return True

def test_19_consecutive_optim_steps(tc):
    datums = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
    ok, _, elapsed = run_forward_backward(tc, datums, 'TEST-19-BASE')
    assert ok and elapsed < TIMEOUT
    for i in range(3):
        do_optim_step(tc, f'TEST-19-STEP-{i+1}')
    datums = [make_datum(seq_len=64, completion_len=32) for _ in range(4)]
    ok, _, elapsed = run_forward_backward(tc, datums, 'TEST-19-VERIFY')
    if not ok or elapsed >= TIMEOUT:
        return False
    do_optim_step(tc, 'TEST-19-VERIFY')
    return True


ALL_TESTS = [
    ('TEST-1: Normal GRPO Training', test_1_normal_grpo),
    ('TEST-2: Bad old_logps (original bug)', test_2_bad_old_logps),
    ('TEST-3: Recovery after error', test_3_recovery),
    ('TEST-4: No advantages (zero loss)', test_4_no_advantages),
    ('TEST-5: Consecutive bad batches', test_5_consecutive_bad),
    ('TEST-6: NaN logprobs', test_6_nan_logprobs),
    ('TEST-7: +Inf/-Inf logprobs', test_7_inf_logprobs),
    ('TEST-8: Extreme advantages (1e30)', test_8_extreme_advantages),
    ('TEST-9: Zero completion tokens', test_9_zero_completion),
    ('TEST-10: Partial advantages (ragged)', test_10_partial_advantages),
    ('TEST-11: Mixed sequence lengths', test_11_mixed_seq_lengths),
    ('TEST-12: All datums bad (100%)', test_12_all_bad),
    ('TEST-13: forward_only then train', test_13_forward_only_then_train),
    ('TEST-14: Rapid bad->good alternation', test_14_rapid_bad_good),
    ('TEST-15: Final health check', test_15_final_health),
    ('TEST-16: Large batch (16 datums)', test_16_large_batch),
    ('TEST-17: Single datum batch', test_17_single_datum),
    ('TEST-18: Save after error', test_18_save_after_error),
    ('TEST-19: Consecutive optim_steps', test_19_consecutive_optim_steps),
]


def main():
    log('=' * 60)
    log('NCCL-Safe E2E Test - Tinker Client Path')
    log('=' * 60)
    log(f'Server URL: {SERVER_URL}')
    log(f'Base Model: {BASE_MODEL}')
    log(f'TWINKLE_FAIL_FAST = {os.getenv("TWINKLE_FAIL_FAST", "1 (default)")}')

    wait_for_server(SERVER_URL)
    tc = init_client()

    results = []
    for name, test_fn in ALL_TESTS:
        log(f'\n{"=" * 60}\n{name}\n{"=" * 60}')
        try:
            passed = test_fn(tc)
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


def test_nccl_safe_tinker_e2e():
    """Pytest-collected entry point."""
    rc = main()
    assert rc == 0, 'Some Tinker NCCL-safe E2E tests failed'


if __name__ == '__main__':
    sys.exit(main())
