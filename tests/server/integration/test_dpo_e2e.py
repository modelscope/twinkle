# Copyright (c) ModelScope Contributors. All rights reserved.
"""DPO (Direct Preference Optimization) E2E integration tests.

Tests DPO training across all 4 combinations:
  - Twinkle client x (transformers | megatron)
  - Tinker client x (transformers | megatron)

Backend selection via env var TWINKLE_TEST_BACKEND (default: transformers).

## How to run

    # Start server
    python tests/server/start_e2e_server.py --config tests/server/config/server_config_4b_e2e.yaml

    # Run DPO tests
    TWINKLE_TEST_GPU_E2E=1 TWINKLE_TEST_BACKEND=transformers pytest tests/server/integration/test_dpo_e2e.py -v
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

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
    BASE_MODEL,
    BASE_URL,
    GRADIENT_ACCUMULATION_STEPS,
    MODEL_ID,
    TIMEOUT,
    assert_metrics_valid,
    assert_no_timeout,
    convert_tensors,
    create_dpo_dataset,
    create_tinker_training_client,
    create_twinkle_dpo_model,
    get_backend,
    init_twinkle_client_session,
    log,
    prepare_dpo_batch,
    wait_for_server,
)

# ── Configuration ──
DPO_TRAIN_STEPS = 8
DPO_BETA = 0.1
DPO_SFT_WEIGHT = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Test: DPO via Twinkle client
# ═══════════════════════════════════════════════════════════════════════════

def test_dpo_twinkle():
    """DPO training via Twinkle client (MultiLoraTransformersModel).

    Flow per step:
      1. forward_only (reference, disable_lora=True) -> ref_outputs
      2. forward_backward (DPO loss with ref_outputs)
      3. clip_grad_and_step

    Pass criteria:
    - All steps complete without timeout (< 120s each)
    - DPO metrics are valid (non-NaN/Inf)
    - No NCCL hang
    """
    import torch
    from twinkle.dataloader import DataLoader

    backend = get_backend()
    log(f'=== test_dpo_twinkle [backend={backend}] ===')

    wait_for_server()
    init_twinkle_client_session()

    # Setup
    dataset = create_dpo_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    model = create_twinkle_dpo_model()

    log(f'Dataset: {len(dataset)} samples, DPO training {DPO_TRAIN_STEPS} steps')

    # Training loop
    losses = []
    reward_margins = []
    step = 0
    for batch in dataloader:
        if step >= DPO_TRAIN_STEPS:
            break

        # Convert tensors for serialization
        convert_tensors(batch)

        # Interleave positive/negative pairs
        dpo_batch = prepare_dpo_batch(batch)

        # Step 1: Reference forward (base model, no LoRA)
        log(f'[step {step + 1}] forward_only (reference)...')
        t0 = time.time()
        ref_outputs = model.forward_only(inputs=dpo_batch, disable_lora=True)
        elapsed_fo = time.time() - t0
        assert_no_timeout(elapsed_fo, f'dpo_twinkle forward_only step {step}')
        log(f'[step {step + 1}] forward_only OK ({elapsed_fo:.1f}s)')

        # Step 2: DPO forward_backward with ref_outputs
        log(f'[step {step + 1}] forward_backward (DPO)...')
        t1 = time.time()
        model.forward_backward(inputs=dpo_batch, ref_outputs=ref_outputs.result)
        elapsed_fb = time.time() - t1
        assert_no_timeout(elapsed_fb, f'dpo_twinkle forward_backward step {step}')
        log(f'[step {step + 1}] forward_backward OK ({elapsed_fb:.1f}s)')

        # Step 3: Optimizer step
        model.clip_grad_and_step()

        # Log metrics every GA steps
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            metrics = model.calculate_metric(is_training=True)
            if hasattr(metrics, 'result'):
                assert_metrics_valid(metrics.result, f'dpo_twinkle step {step}')
                # Track DPO loss and rewards
                result = metrics.result
                if isinstance(result, dict):
                    loss_val = result.get('loss')
                    if loss_val is not None:
                        losses.append(float(loss_val))
                    reward_margin = result.get('rewards/margins')
                    if reward_margin is not None:
                        reward_margins.append(float(reward_margin))

        step += 1

    assert step == DPO_TRAIN_STEPS, f'Expected {DPO_TRAIN_STEPS} steps, completed {step}'

    # Verify DPO loss decreases
    if len(losses) >= 3:
        log(f'DPO losses: {["{:.4f}".format(l) for l in losses]}')
        assert losses[-1] < losses[0], (
            f'[dpo_twinkle] DPO loss did NOT decrease: first={losses[0]:.4f} last={losses[-1]:.4f}')
        log(f'[dpo_twinkle] Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}')

    # Verify reward margins increase (DPO learns to prefer chosen)
    if len(reward_margins) >= 3:
        log(f'Reward margins: {["{:.4f}".format(r) for r in reward_margins]}')
        assert reward_margins[-1] > reward_margins[0], (
            f'[dpo_twinkle] Reward margins did NOT increase: first={reward_margins[0]:.4f} last={reward_margins[-1]:.4f}')
        log(f'[dpo_twinkle] Reward margins increased: {reward_margins[0]:.4f} -> {reward_margins[-1]:.4f}')

    log(f'test_dpo_twinkle PASSED (backend={backend})')


# ═══════════════════════════════════════════════════════════════════════════
# Test: DPO via Tinker client
# ═══════════════════════════════════════════════════════════════════════════

def test_dpo_tinker():
    """DPO training via Tinker client (ServiceClient + forward/forward_backward).

    Flow per step:
      1. forward (cross_entropy, disable_lora=True) -> ref logps
      2. Attach ref_logps to datums
      3. forward_backward (importance_sampling) -> DPO loss
      4. optim_step

    Pass criteria:
    - All steps complete without timeout (< 120s each)
    - Metrics are valid
    - No NCCL hang
    """
    from tinker import types
    from twinkle.dataloader import DataLoader
    from twinkle.server.common import input_feature_to_datum

    backend = get_backend()
    log(f'=== test_dpo_tinker [backend={backend}] ===')

    wait_for_server()

    # Setup
    dataset = create_dpo_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    training_client = create_tinker_training_client(rank=8)

    log(f'Dataset: {len(dataset)} samples, DPO training {DPO_TRAIN_STEPS} steps')

    # Training loop
    losses = []
    reward_margins = []
    step = 0
    for batch in dataloader:
        if step >= DPO_TRAIN_STEPS:
            break

        # Convert tensors
        convert_tensors(batch)

        # Interleave positive/negative pairs
        dpo_batch = prepare_dpo_batch(batch)

        # Convert to Tinker Datums
        input_datums = [input_feature_to_datum(row) for row in dpo_batch]

        # Step A: Reference forward (base model, disable_lora=True)
        log(f'[step {step + 1}] forward (reference, disable_lora)...')
        t0 = time.time()
        ref_result = training_client.forward(
            input_datums,
            'cross_entropy',
            loss_fn_config={'disable_lora': True},
        ).result()
        elapsed_ref = time.time() - t0
        assert_no_timeout(elapsed_ref, f'dpo_tinker reference step {step}')
        log(f'[step {step + 1}] reference forward OK ({elapsed_ref:.1f}s)')

        # Step B: Attach ref_logps to datums
        for datum, ref_out in zip(input_datums, ref_result.loss_fn_outputs):
            ref_logprobs_np = np.array(ref_out['logprobs'].tolist(), dtype=np.float32)
            datum.loss_fn_inputs['ref_logps'] = types.TensorData.from_numpy(ref_logprobs_np)

        # Step C: DPO forward_backward
        log(f'[step {step + 1}] forward_backward (DPO)...')
        t1 = time.time()
        fwdbwd_result = training_client.forward_backward(
            input_datums,
            'importance_sampling',
            loss_fn_config={
                'dpo_beta': DPO_BETA,
                'dpo_sft_weight': DPO_SFT_WEIGHT,
            },
        ).result()
        elapsed_fb = time.time() - t1
        assert_no_timeout(elapsed_fb, f'dpo_tinker forward_backward step {step}')
        log(f'[step {step + 1}] forward_backward OK ({elapsed_fb:.1f}s)')

        # Step D: Optimizer step
        optim_result = training_client.optim_step(
            types.AdamParams(learning_rate=1e-4)
        ).result()

        if optim_result.metrics:
            assert_metrics_valid(optim_result.metrics, f'dpo_tinker step {step}')
            log(f'[step {step + 1}] metrics={optim_result.metrics}')
            # Track loss and reward margins
            loss_val = optim_result.metrics.get('loss')
            if loss_val is not None:
                losses.append(float(loss_val))
            reward_margin = optim_result.metrics.get('rewards/margins')
            if reward_margin is not None:
                reward_margins.append(float(reward_margin))

        step += 1

    assert step == DPO_TRAIN_STEPS, f'Expected {DPO_TRAIN_STEPS} steps, completed {step}'

    # Verify DPO loss decreases
    if len(losses) >= 3:
        log(f'DPO losses: {["{:.4f}".format(l) for l in losses]}')
        assert losses[-1] < losses[0], (
            f'[dpo_tinker] DPO loss did NOT decrease: first={losses[0]:.4f} last={losses[-1]:.4f}')
        log(f'[dpo_tinker] Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}')

    # Verify reward margins increase
    if len(reward_margins) >= 3:
        log(f'Reward margins: {["{:.4f}".format(r) for r in reward_margins]}')
        assert reward_margins[-1] > reward_margins[0], (
            f'[dpo_tinker] Reward margins did NOT increase: first={reward_margins[0]:.4f} last={reward_margins[-1]:.4f}')
        log(f'[dpo_tinker] Reward margins increased: {reward_margins[0]:.4f} -> {reward_margins[-1]:.4f}')

    log(f'test_dpo_tinker PASSED (backend={backend})')


# ── Direct execution ──

def main() -> int:
    log('Running DPO E2E tests directly...')
    try:
        test_dpo_twinkle()
        test_dpo_tinker()
        log('ALL DPO TESTS PASSED')
        return 0
    except Exception as e:
        log(f'FAILED: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
