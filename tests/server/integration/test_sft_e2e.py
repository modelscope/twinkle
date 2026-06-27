# Copyright (c) ModelScope Contributors. All rights reserved.
"""SFT (Supervised Fine-Tuning) E2E integration tests.

Tests SFT training across all 4 combinations:
  - Twinkle client x (transformers | megatron)
  - Tinker client x (transformers | megatron)

Backend selection via env var TWINKLE_TEST_BACKEND (default: transformers).

## How to run

    # Start server (transformers or megatron)
    python tests/server/start_e2e_server.py --config tests/server/config/server_config_4b_e2e.yaml

    # Run SFT tests
    TWINKLE_TEST_GPU_E2E=1 TWINKLE_TEST_BACKEND=transformers pytest tests/server/integration/test_sft_e2e.py -v
"""
from __future__ import annotations

import os
import sys
import time

# Ensure project root is in sys.path for both pytest and direct execution
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Set TWINKLE_TEST_GPU_E2E=1 to run real GPU E2E tests (requires running server)',
)

from tests.server.integration.e2e_helpers import (
    BASE_URL,
    GRADIENT_ACCUMULATION_STEPS,
    TIMEOUT,
    assert_loss_decreases,
    assert_no_timeout,
    convert_tensors,
    create_sft_dataset,
    create_tinker_training_client,
    create_twinkle_sft_model,
    get_backend,
    init_twinkle_client_session,
    log,
    wait_for_server,
)

# ── Configuration ──
SFT_TRAIN_STEPS = 20  # 20 steps ensures enough training for both backends


# ═══════════════════════════════════════════════════════════════════════════
# Test: SFT via Twinkle client
# ═══════════════════════════════════════════════════════════════════════════

def test_sft_twinkle():
    """SFT training via Twinkle client (MultiLoraTransformersModel).

    Pass criteria:
    - Training completes 10 steps without timeout
    - Loss shows downward trend (last_3_avg < first_3_avg)
    """
    backend = get_backend()
    log(f'=== test_sft_twinkle [backend={backend}] ===')

    wait_for_server()
    init_twinkle_client_session()

    # Setup
    from twinkle.dataloader import DataLoader

    dataset = create_sft_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    model = create_twinkle_sft_model()

    log(f'Dataset: {len(dataset)} samples, {len(dataloader)} batches')
    log(f'Training {SFT_TRAIN_STEPS} steps (GA={GRADIENT_ACCUMULATION_STEPS})')

    # Training loop
    losses = []
    for step, batch in enumerate(dataloader):
        if step >= SFT_TRAIN_STEPS:
            break

        t0 = time.time()
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        elapsed = time.time() - t0
        assert_no_timeout(elapsed, f'sft_twinkle step {step}')

        # Log metric every GA steps
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            metric = model.calculate_metric(is_training=True)
            try:
                loss = float(metric.result.get('loss')) if hasattr(metric.result, 'get') else float(
                    metric.result['loss'])
            except Exception:
                loss = float('nan')
            losses.append(loss)
            log(f'[step {step + 1}] loss={loss:.4f} ({elapsed:.1f}s)')

    # Assertions — both backends should report real loss via calculate_metric
    assert len(losses) >= 4, f'Expected at least 4 logged losses, got {len(losses)}'
    assert_loss_decreases(losses, 'sft_twinkle')
    log(f'test_sft_twinkle PASSED (backend={backend})')


# ═══════════════════════════════════════════════════════════════════════════
# Test: SFT via Tinker client
# ═══════════════════════════════════════════════════════════════════════════

def test_sft_tinker():
    """SFT training via Tinker client (ServiceClient + forward_backward).

    Pass criteria:
    - Training completes 10 steps without timeout
    - Loss shows downward trend (last_3_avg < first_3_avg)
    """
    from tinker import types
    from twinkle.dataloader import DataLoader
    from twinkle.server.common import input_feature_to_datum

    backend = get_backend()
    log(f'=== test_sft_tinker [backend={backend}] ===')

    wait_for_server()

    # Setup
    dataset = create_sft_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    training_client = create_tinker_training_client(rank=16)

    log(f'Dataset: {len(dataset)} samples, {len(dataloader)} batches')
    log(f'Training {SFT_TRAIN_STEPS} steps')

    # Training loop
    losses = []
    for step, batch in enumerate(dataloader):
        if step >= SFT_TRAIN_STEPS:
            break

        # Convert batch to Tinker Datums
        input_datums = [input_feature_to_datum(input_feature) for input_feature in batch]

        # Forward-backward
        t0 = time.time()
        fwdbwd_result = training_client.forward_backward(input_datums, 'cross_entropy').result()
        elapsed_fb = time.time() - t0
        assert_no_timeout(elapsed_fb, f'sft_tinker forward_backward step {step}')

        # Optimizer step
        optim_result = training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        elapsed_total = time.time() - t0
        assert_no_timeout(elapsed_total, f'sft_tinker total step {step}')

        # Compute loss from logprobs
        try:
            logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
            weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in input_datums])
            loss = float(-np.dot(logprobs, weights) / max(weights.sum(), 1e-8))
        except Exception:
            loss = float('nan')
        losses.append(loss)
        log(f'[step {step + 1}] loss={loss:.4f} ({elapsed_total:.1f}s)')

    # Assertions
    assert len(losses) >= 4, f'Expected at least 4 logged losses, got {len(losses)}'
    assert_loss_decreases(losses, 'sft_tinker')
    log(f'test_sft_tinker PASSED (backend={backend})')


# ── Direct execution ──

def main() -> int:
    log('Running SFT E2E tests directly...')
    try:
        test_sft_twinkle()
        test_sft_tinker()
        log('ALL SFT TESTS PASSED')
        return 0
    except Exception as e:
        log(f'FAILED: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
