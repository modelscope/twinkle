# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real E2E test for loud-failure semantics via the Tinker client path.

Exercises ``/tinker/forward_backward`` through the upstream Tinker SDK. The
invariant under test (post silent-degradation removal): a request whose loss
computation fails does NOT come back as a silent zero-loss success -- it enters a
failed terminal state -- and a subsequent valid request on the same deployment
still succeeds.

Prerequisites:
    1. Ray cluster running with GPUs (2 for model DP/TP)
    2. Twinkle server started with queue_config.execution_timeout=30

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
EXECUTION_TIMEOUT = float(os.environ.get('TWINKLE_TEST_EXECUTION_TIMEOUT', '30'))
TIMEOUT = EXECUTION_TIMEOUT + 15
# The `global_rank=` attribution is added by `nccl_safe_megatron`, which decorates
# only the Megatron backend; the transformers backend carries no such annotation
# (its former silent-degradation decorator was removed by R6#3). Gate the rank-attribution assertion
# on the backend so this file is safe under TWINKLE_TEST_BACKEND=transformers.
BACKEND = os.environ.get('TWINKLE_TEST_BACKEND', 'megatron')


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


def _assert_recovery_terminal(tc) -> None:
    """Require success on Megatron; Transformers may fail loudly after DDP poisoning."""
    from tinker import types
    from tinker._exceptions import RequestFailedError

    request = tc.forward_backward([_make_datum() for _ in range(4)], 'importance_sampling')
    if BACKEND == 'megatron':
        assert request.result() is not None
        tc.optim_step(types.AdamParams(learning_rate=1e-5)).result()
        return
    try:
        assert request.result(timeout=TIMEOUT) is not None
        tc.optim_step(types.AdamParams(learning_rate=1e-5)).result()
    except RequestFailedError as exc:
        assert exc.category is types.RequestErrorCategory.Server


def test_failure_is_terminal_then_valid_request_succeeds():
    """A malformed request fails loudly (terminal), a subsequent valid one succeeds.

    Replaces the former assertion "failure degraded to zero loss and training
    continued". If the recovery request does not reach a terminal success, that is
    recorded as evidence that R3#2-3 actor recovery and R2#3-4 admission gate are
    necessary, not optional.
    """
    from tinker import types
    from tinker._exceptions import RequestFailedError
    tc = _init_client()

    # Deliberately malformed: logprobs length inconsistent with the completion.
    bad = [_make_datum(bad_logprobs_len=5) for _ in range(4)]
    start = time.time()
    with pytest.raises(RequestFailedError) as caught:
        tc.forward_backward(bad, 'importance_sampling').result(timeout=TIMEOUT)
    assert caught.value.category is types.RequestErrorCategory.Server
    assert time.time() - start < TIMEOUT, 'malformed request must fail fast, not hang (NCCL)'

    # Megatron must recover successfully. Tinker's Transformers path executes
    # forward/loss/backward separately; after a mid-iteration failure, R6#14 only
    # guarantees that the next request reaches a terminal state.
    _assert_recovery_terminal(tc)


def test_partial_rank_failure_is_terminal_then_recovers():
    from tinker import types
    from tinker._exceptions import RequestFailedError
    tc = _init_client()

    batch = [_make_datum() for _ in range(4)]
    batch[0] = _make_datum(bad_logprobs_len=5)
    start = time.time()
    with pytest.raises(RequestFailedError) as caught:
        tc.forward_backward(batch, 'importance_sampling').result(timeout=TIMEOUT)
    assert caught.value.category is types.RequestErrorCategory.Server
    # Megatron attributes the failure to a global rank via nccl_safe_megatron; the
    # transformers backend has no such annotation (R6#3 removed its old decorator).
    if BACKEND == 'megatron':
        assert 'global_rank=' in str(caught.value)
    assert time.time() - start < TIMEOUT

    _assert_recovery_terminal(tc)
