# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real E2E test for loud-failure semantics via the Tinker client path.

Exercises ``/tinker/forward_backward`` through the upstream Tinker SDK. The
invariant under test (post silent-degradation removal): a request whose loss
computation fails does NOT come back as a silent zero-loss success -- it enters a
failed terminal state -- and a subsequent valid request on the same deployment
still succeeds.

Prerequisites:
    1. Ray cluster running with GPUs (2 for model DP/TP)
    2. Twinkle server started (no fault-tolerance env switch exists any more)

Usage (pytest, requires TWINKLE_TEST_GPU_E2E=1):
    TWINKLE_TEST_GPU_E2E=1 pytest tests/server/integration/test_nccl_safe_tinker_e2e.py -v
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Set TWINKLE_TEST_GPU_E2E=1 to run real GPU E2E tests (requires running server)',
)

BASE_MODEL = 'Qwen/Qwen3.5-4B'
SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:9000')
TIMEOUT = 120


def _init_client():
    os.environ['TINKER_BASE_URL'] = SERVER_URL
    os.environ['TWINKLE_SERVER_TOKEN'] = 'EMPTY_TOKEN'
    from twinkle_client import init_tinker_client
    init_tinker_client()
    from tinker import ServiceClient
    return ServiceClient().create_lora_training_client(base_model=BASE_MODEL, rank=16)


def _make_datum(seq_len=64, completion_len=32, *, bad_logprobs_len=None):
    from tinker import types
    prompt_len = seq_len - completion_len
    input_tokens = list(range(1, seq_len + 1))
    target_tokens = [0] * prompt_len + list(range(100, 100 + completion_len))
    weights = [0] * prompt_len + [1] * completion_len
    n = bad_logprobs_len if bad_logprobs_len is not None else completion_len
    padded_logprobs = [0.0] * prompt_len + np.random.randn(n).astype(np.float32).tolist()
    advantage = float(np.random.randn())
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            'target_tokens': target_tokens,
            'weights': weights,
            'logprobs': types.TensorData.from_numpy(np.array(padded_logprobs, dtype=np.float32)),
            'advantages': types.TensorData.from_numpy(
                np.array([0.0] * prompt_len + [advantage] * completion_len, dtype=np.float32)),
        },
    )


def test_failure_is_terminal_then_valid_request_succeeds():
    """A malformed request fails loudly (terminal), a subsequent valid one succeeds.

    Replaces the former assertion "failure degraded to zero loss and training
    continued". If the recovery request does not reach a terminal success, that is
    recorded as evidence that R3#2-3 actor recovery and R2#3-4 admission gate are
    necessary, not optional.
    """
    from tinker import types
    tc = _init_client()

    # Deliberately malformed: logprobs length inconsistent with the completion.
    bad = [_make_datum(bad_logprobs_len=5) for _ in range(4)]
    start = time.time()
    with pytest.raises(Exception):  # RequestFailedError or a raised failed terminal
        tc.forward_backward(bad, 'importance_sampling').result()
    assert time.time() - start < TIMEOUT, 'malformed request must fail fast, not hang (NCCL)'

    # Recovery: a subsequent valid request on the same deployment must succeed.
    good = [_make_datum() for _ in range(4)]
    result = tc.forward_backward(good, 'importance_sampling').result()
    assert result is not None
    tc.optim_step(types.AdamParams(learning_rate=1e-5)).result()
