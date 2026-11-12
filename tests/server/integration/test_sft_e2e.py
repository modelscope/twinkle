# Copyright (c) ModelScope Contributors. All rights reserved.
"""SFT (Supervised Fine-Tuning) E2E integration tests.

Tests SFT training across all 4 combinations:
  - Twinkle client x (transformers | megatron)
  - Tinker client x (transformers | megatron)

Each test also verifies save-LoRA + resume-training succeeds: after the training
loop it saves a checkpoint (Twinkle ``model.save``; Tinker ``save_state``), resumes
(Twinkle ``resume_from_checkpoint``; Tinker
``create_training_client_from_state_with_optimizer``), and runs a few more steps
that must complete without timeout.

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
    init_tinker_client_session,
    init_twinkle_client_session,
    log,
    wait_for_server,
)

# ── Configuration ──
SFT_TRAIN_STEPS = 20  # 20 steps ensures enough training for both backends
SFT_RESUME_STEPS = 3  # post-resume steps that must run without timeout


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

    # ── Save LoRA + resume training (must succeed) ──
    save_resp = model.save(
        name='sft-twinkle-resume',
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )
    ckpt = save_resp.twinkle_path
    assert ckpt, 'save() did not return a twinkle_path'
    log(f'saved LoRA checkpoint: {ckpt}')

    progress = model.resume_from_checkpoint(ckpt)
    log(f'resumed from checkpoint: {progress}')

    resume_loader = DataLoader(dataset=create_sft_dataset(), batch_size=4)
    resumed = 0
    for step, batch in enumerate(resume_loader):
        if step >= SFT_RESUME_STEPS:
            break
        t0 = time.time()
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        assert_no_timeout(time.time() - t0, f'sft_twinkle resume step {step}')
        resumed += 1
    assert resumed == SFT_RESUME_STEPS, f'expected {SFT_RESUME_STEPS} post-resume steps, ran {resumed}'
    log(f'test_sft_twinkle PASSED (backend={backend}) [+save LoRA +resume]')


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
    from twinkle.server.model.tinker_datum import input_feature_to_datum

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

    # ── Save state + resume training (must succeed) ──
    save_result = training_client.save_state('sft-tinker-resume').result()
    state_path = save_result.path
    assert state_path, 'save_state() did not return a path'
    log(f'saved tinker state: {state_path}')

    # Resume restores both weights and optimizer state into a fresh client.
    service_client = init_tinker_client_session()
    resumed_client = service_client.create_training_client_from_state_with_optimizer(path=state_path)
    log('resumed tinker training client from saved state')

    resume_loader = DataLoader(dataset=create_sft_dataset(), batch_size=4)
    resumed = 0
    for step, batch in enumerate(resume_loader):
        if step >= SFT_RESUME_STEPS:
            break
        input_datums = [input_feature_to_datum(input_feature) for input_feature in batch]
        t0 = time.time()
        resumed_client.forward_backward(input_datums, 'cross_entropy').result()
        resumed_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        assert_no_timeout(time.time() - t0, f'sft_tinker resume step {step}')
        resumed += 1
    assert resumed == SFT_RESUME_STEPS, f'expected {SFT_RESUME_STEPS} post-resume steps, ran {resumed}'
    log(f'test_sft_tinker PASSED (backend={backend}) [+save state +resume]')


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
