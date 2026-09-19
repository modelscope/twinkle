# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real E2E test for loud-failure semantics via the Twinkle-native client path.

Exercises ``/twinkle/forward_backward`` through the Twinkle client. The invariant
under test (post silent-degradation removal): a request whose loss computation fails
does NOT come back as a silent zero-loss success -- it fails loudly -- and a
subsequent valid request on the same deployment still succeeds.

Prerequisites:
    1. Ray cluster running with GPUs (2 for model DP/TP)
    2. Twinkle server started with queue_config.execution_timeout=30

Usage (pytest, requires TWINKLE_TEST_GPU_E2E=1):
    TWINKLE_TEST_GPU_E2E=1 pytest tests/server/integration/test_nccl_safe_twinkle_e2e.py -v
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
ADAPTER_NAME = 'loud-failure-test'
# The `global_rank=` attribution is added by `nccl_safe_megatron`, which decorates
# only the Megatron backend; the transformers backend's forward_backward carries no
# such annotation (its former silent-degradation decorator was removed by R6#3). Gate the
# rank-attribution assertion on the backend so this file is safe to run under the
# integration-e2e SKILL's TWINKLE_TEST_BACKEND=transformers path.
BACKEND = os.environ.get('TWINKLE_TEST_BACKEND', 'megatron')


def _init_client():
    from peft import LoraConfig
    from twinkle_client import init_twinkle_client
    from twinkle_client.model import MultiLoraTransformersModel

    init_twinkle_client(base_url=SERVER_URL, api_key='EMPTY_TOKEN')
    model = MultiLoraTransformersModel(model_id=f'ms://{BASE_MODEL}')
    model.add_adapter_to_model(
        adapter_name=ADAPTER_NAME,
        config=LoraConfig(r=16, target_modules=['q_proj', 'v_proj']),
        # GA>=2 (repo convention, see e2e_helpers): with GA=1 every backward syncs
        # DDP immediately, so a mid-iteration failure can leave the reducer
        # half-finished and poison the next request. GA=2 runs accumulation steps
        # under no_sync, keeping the recovery request clean.
        gradient_accumulation_steps=2,
    )
    model.set_loss('GRPOLoss', init_args={'epsilon': 0.2})
    model.set_optimizer('Adam', lr=1e-5)
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    return model


def _make_inputs(batch_size=4, seq_len=64, completion_len=32, *, bad_old_logps_len=None):
    prompt_len = seq_len - completion_len
    features, old_logps, advantages = [], [], []
    for _ in range(batch_size):
        features.append({
            'input_ids': list(range(1, seq_len + 1)),
            'labels': [-100] * prompt_len + list(range(100, 100 + completion_len)),
            'attention_mask': [1] * seq_len,
            'position_ids': list(range(seq_len)),
        })
        n = bad_old_logps_len if bad_old_logps_len is not None else completion_len
        old_logps.append(np.random.randn(n).tolist())
        advantages.append(float(np.random.randn()))
    return features, old_logps, advantages


def test_failure_is_terminal_then_valid_request_succeeds():
    """A malformed request fails loudly, a subsequent valid one succeeds.

    Replaces the former assertion "failure degraded to zero loss and training
    continued". If the recovery request does not reach a terminal success, that is
    recorded as evidence that R3#2-3 actor recovery and R2#3-4 admission gate are
    necessary, not optional.
    """
    model = _init_client()

    bad_features, bad_old_logps, bad_adv = _make_inputs(bad_old_logps_len=5)
    start = time.time()
    with pytest.raises(Exception) as caught:
        model.forward_backward(
            inputs=bad_features, adapter_name=ADAPTER_NAME, old_logps=bad_old_logps, advantages=bad_adv)
    message = str(caught.value)
    # The failure must be loud and descriptive (not a silent zero-loss success):
    # the deliberate old_logps/completion length mismatch surfaces on both backends.
    assert 'mismatch' in message, message
    # Megatron additionally attributes the failure to a global rank via nccl_safe_megatron.
    if BACKEND == 'megatron':
        assert 'global_rank=' in message, message
    assert time.time() - start < TIMEOUT, 'malformed request must fail fast, not hang (NCCL)'

    good_features, good_old_logps, good_adv = _make_inputs()
    result = model.forward_backward(
        inputs=good_features, adapter_name=ADAPTER_NAME, old_logps=good_old_logps, advantages=good_adv)
    assert result is not None
