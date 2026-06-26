# Copyright (c) ModelScope Contributors. All rights reserved.
"""GRPO (Group Relative Policy Optimization) E2E integration tests.

Tests GRPO training across all 4 combinations:
  - Twinkle client x (transformers | megatron)
  - Tinker client x (transformers | megatron)

Backend selection via env var TWINKLE_TEST_BACKEND (default: transformers).
Requires sampler service to be running.

## How to run

    # Start server (must include sampler)
    python tests/server/start_e2e_server.py --config tests/server/config/server_config_4b_e2e.yaml

    # Run GRPO tests
    TWINKLE_TEST_GPU_E2E=1 TWINKLE_TEST_BACKEND=transformers pytest tests/server/integration/test_grpo_e2e.py -v
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

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
    create_grpo_dataset,
    create_tinker_training_client,
    create_twinkle_grpo_model,
    create_twinkle_sampler,
    get_backend,
    init_twinkle_client_session,
    log,
    wait_for_server,
)

# ── Configuration ──
GRPO_TRAIN_STEPS = 2
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 512
LEARNING_RATE = 2e-5
TEMPERATURE = 1.0

SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                 'and put your final answer within \\boxed{}.')


# ═══════════════════════════════════════════════════════════════════════════
# Reward Functions (lightweight versions for E2E testing)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rewards(trajectories: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """Compute accuracy and brevity rewards for GSM8K.

    Returns (total_rewards, accuracy_rewards).
    """
    from twinkle.reward import GSM8KAccuracyReward

    accuracy_reward_fn = GSM8KAccuracyReward()
    accuracy_rewards = accuracy_reward_fn(trajectories)

    # Simple brevity reward
    brevity_rewards = []
    for traj in trajectories:
        messages = traj.get('messages', [])
        completion = ''
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                completion = msg.get('content', '')
                break
        has_answer = bool(
            re.search(r'\\boxed\{[^}]+\}', completion)
            or re.search(r'####\s*[\-\d,\.]+', completion)
        )
        if not has_answer:
            brevity_rewards.append(0.0)
        else:
            length = len(completion)
            brevity_rewards.append(max(0.0, 1.0 - max(0, length - 200) / 3000))

    total_rewards = [a + b for a, b in zip(accuracy_rewards, brevity_rewards)]
    return total_rewards, accuracy_rewards


# ═══════════════════════════════════════════════════════════════════════════
# Test: GRPO via Twinkle client
# ═══════════════════════════════════════════════════════════════════════════

def test_grpo_twinkle():
    """GRPO training via Twinkle client (model.save + sampler + forward_backward).

    Flow per step:
      1. model.save(is_sampler=True) -> adapter_uri
      2. sampler.sample(inputs, adapter_uri, num_samples=N)
      3. Compute rewards + advantages
      4. model.forward_backward(inputs, advantages, old_logps)
      5. model.clip_grad_and_step()

    Pass criteria:
    - Sampling returns non-empty completions
    - forward_backward completes without timeout
    - Metrics are valid (non-NaN/Inf)
    - No NCCL hang
    """
    from twinkle.advantage import GRPOAdvantage
    from twinkle.dataloader import DataLoader

    backend = get_backend()
    log(f'=== test_grpo_twinkle [backend={backend}] ===')

    wait_for_server()
    init_twinkle_client_session()

    # Setup
    dataset = create_grpo_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)
    model = create_twinkle_grpo_model()
    sampler = create_twinkle_sampler()
    advantage_fn = GRPOAdvantage()

    sampling_params = {
        'max_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'top_p': 0.95,
        'num_samples': NUM_GENERATIONS,
        'logprobs': 1,
    }

    log(f'Dataset: {len(dataset)} samples, GRPO training {GRPO_TRAIN_STEPS} steps')
    log(f'NUM_GENERATIONS={NUM_GENERATIONS}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}')

    # Training loop
    current_adapter_uri = None
    step = 0

    for batch in dataloader:
        if step >= GRPO_TRAIN_STEPS:
            break

        prompts = batch if isinstance(batch, list) else [batch]

        # Step 1: Save weights for sampler
        log(f'[step {step + 1}] Saving weights for sampler...')
        t0 = time.time()
        save_result = model.save(name='grpo-e2e-weights', save_optimizer=False, is_sampler=True)
        current_adapter_uri = save_result.twinkle_path
        elapsed_save = time.time() - t0
        log(f'[step {step + 1}] Weights saved ({elapsed_save:.1f}s): {current_adapter_uri}')

        # Step 2: Sample completions
        log(f'[step {step + 1}] Sampling {len(prompts)} prompts x {NUM_GENERATIONS} generations...')
        t1 = time.time()
        sample_responses = sampler.sample(
            inputs=prompts,
            sampling_params=sampling_params,
            adapter_uri=current_adapter_uri,
        )
        elapsed_sample = time.time() - t1
        assert_no_timeout(elapsed_sample, f'grpo_twinkle sampling step {step}')

        # Collect sequences
        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        assert len(all_input_data) > 0, f'[step {step + 1}] Sampling returned no completions!'
        log(f'[step {step + 1}] Got {len(all_input_data)} completions ({elapsed_sample:.1f}s)')

        # Step 3: Compute rewards + advantages
        total_rewards, accuracy_rewards = compute_rewards(all_input_data)
        advantages = advantage_fn(
            total_rewards,
            num_generations=NUM_GENERATIONS,
            scale='group',
        ).tolist()

        # Skip if all advantages are zero (no learning signal)
        if all(abs(a) < 1e-8 for a in advantages):
            log(f'[step {step + 1}] All advantages zero, skipping (still counts as success)')
            step += 1
            continue

        # Step 4: forward_backward with GRPO loss
        log(f'[step {step + 1}] forward_backward (GRPO)...')
        t2 = time.time()
        model.forward_backward(
            inputs=all_input_data,
            advantages=advantages,
            old_logps=all_old_logps,
        )
        elapsed_fb = time.time() - t2
        assert_no_timeout(elapsed_fb, f'grpo_twinkle forward_backward step {step}')
        log(f'[step {step + 1}] forward_backward OK ({elapsed_fb:.1f}s)')

        # Step 5: Optimizer step
        model.clip_grad_and_step()

        # Log metrics
        metrics = model.calculate_metric(is_training=True)
        if hasattr(metrics, 'result'):
            assert_metrics_valid(metrics.result, f'grpo_twinkle step {step}')

        step += 1
        log(f'[step {step}] Complete. accuracy_rewards={accuracy_rewards[:4]}')

    assert step >= GRPO_TRAIN_STEPS, f'Expected {GRPO_TRAIN_STEPS} steps, completed {step}'
    log(f'test_grpo_twinkle PASSED (backend={backend})')


# ═══════════════════════════════════════════════════════════════════════════
# Test: GRPO via Tinker client
# ═══════════════════════════════════════════════════════════════════════════

def test_grpo_tinker():
    """GRPO training via Tinker client (save_weights_and_get_sampling_client).

    Flow per step:
      1. save_weights_and_get_sampling_client() -> sampling_client
      2. sampling_client.sample(prompt, params, num_samples=N)
      3. Compute rewards + advantages
      4. Build Datums with logprobs + advantages
      5. forward_backward (importance_sampling)
      6. optim_step

    Pass criteria:
    - Sampling returns non-empty completions
    - forward_backward completes without timeout
    - Metrics are valid
    - No NCCL hang
    """
    from tinker import types
    from twinkle.advantage import GRPOAdvantage
    from twinkle.dataloader import DataLoader
    from twinkle.template import Qwen3_5Template

    backend = get_backend()
    log(f'=== test_grpo_tinker [backend={backend}] ===')

    wait_for_server()

    # Setup
    dataset = create_grpo_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)
    training_client = create_tinker_training_client(rank=8)
    template = Qwen3_5Template(model_id=MODEL_ID)
    advantage_fn = GRPOAdvantage()

    sampling_params = types.SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )

    log(f'Dataset: {len(dataset)} samples, GRPO training {GRPO_TRAIN_STEPS} steps')

    # Training loop
    sampling_client = None
    step = 0

    for batch in dataloader:
        if step >= GRPO_TRAIN_STEPS:
            break

        prompts = batch if isinstance(batch, list) else [batch]

        # Step 1: Save weights and get sampling client
        log(f'[step {step + 1}] Saving weights for sampler...')
        t0 = time.time()
        sampling_client = training_client.save_weights_and_get_sampling_client()
        elapsed_save = time.time() - t0
        log(f'[step {step + 1}] Sampling client ready ({elapsed_save:.1f}s)')

        # Step 2: Sample completions
        log(f'[step {step + 1}] Sampling...')
        t1 = time.time()
        all_sequences = []
        all_user_data = []

        for prompt_feature in prompts:
            input_ids = prompt_feature['input_ids']
            if hasattr(input_ids, 'tolist'):
                input_ids = input_ids.tolist()
            prompt = types.ModelInput.from_ints(input_ids)
            future = sampling_client.sample(
                prompt=prompt,
                sampling_params=sampling_params,
                num_samples=NUM_GENERATIONS,
            )
            result = future.result()
            for _ in range(NUM_GENERATIONS):
                all_user_data.append(prompt_feature.get('user_data', []))
            all_sequences.extend(result.sequences)

        elapsed_sample = time.time() - t1
        assert_no_timeout(elapsed_sample, f'grpo_tinker sampling step {step}')
        assert len(all_sequences) > 0, f'[step {step + 1}] Sampling returned no sequences!'
        log(f'[step {step + 1}] Got {len(all_sequences)} sequences ({elapsed_sample:.1f}s)')

        # Step 3: Build trajectories and compute rewards
        trajectories = []
        completion_lengths = []

        for idx, seq in enumerate(all_sequences):
            decoded_text = template.decode(seq.tokens, skip_special_tokens=True)
            trajectories.append({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': 'Math problem'},
                    {'role': 'assistant', 'content': decoded_text},
                ],
                'user_data': all_user_data[idx],
            })
            completion_lengths.append(len(seq.tokens))

        total_rewards, accuracy_rewards = compute_rewards(trajectories)

        # Step 4: Compute advantages
        advantages = advantage_fn(
            total_rewards,
            num_generations=NUM_GENERATIONS,
            scale='group',
        ).tolist()

        if all(abs(a) < 1e-8 for a in advantages):
            log(f'[step {step + 1}] All advantages zero, skipping')
            step += 1
            continue

        # Step 5: Build training Datums
        training_data = []
        for i, seq in enumerate(all_sequences):
            prompt_feature = prompts[i // NUM_GENERATIONS]
            prompt_ids = prompt_feature['input_ids']
            if hasattr(prompt_ids, 'tolist'):
                prompt_ids = prompt_ids.tolist()

            sampled_tokens = list(seq.tokens)
            logprobs = seq.logprobs if seq.logprobs else [0.0] * len(sampled_tokens)
            advantage = float(advantages[i])

            ob_len = len(prompt_ids) - 1
            input_tokens = prompt_ids + sampled_tokens[:-1]
            target_tokens = [0] * ob_len + sampled_tokens
            weights = [0] * ob_len + [1] * len(sampled_tokens)
            padded_advantages = [0.0] * ob_len + [advantage] * len(sampled_tokens)
            padded_logprobs = [0.0] * ob_len + list(logprobs)

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    'target_tokens': target_tokens,
                    'weights': weights,
                    'logprobs': types.TensorData.from_numpy(np.array(padded_logprobs, dtype=np.float32)),
                    'advantages': types.TensorData.from_numpy(np.array(padded_advantages, dtype=np.float32)),
                },
            )
            training_data.append(datum)

        if not training_data:
            log(f'[step {step + 1}] No training data, skipping')
            step += 1
            continue

        # Step 6: forward_backward with importance_sampling
        log(f'[step {step + 1}] forward_backward ({len(training_data)} datums)...')
        t2 = time.time()
        fwdbwd_result = training_client.forward_backward(training_data, 'importance_sampling').result()
        elapsed_fb = time.time() - t2
        assert_no_timeout(elapsed_fb, f'grpo_tinker forward_backward step {step}')
        log(f'[step {step + 1}] forward_backward OK ({elapsed_fb:.1f}s)')

        # Step 7: Optimizer step
        optim_result = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()
        if optim_result.metrics:
            assert_metrics_valid(optim_result.metrics, f'grpo_tinker step {step}')

        step += 1
        log(f'[step {step}] Complete. accuracy_rewards={accuracy_rewards[:4]}')

    assert step >= GRPO_TRAIN_STEPS, f'Expected {GRPO_TRAIN_STEPS} steps, completed {step}'
    log(f'test_grpo_tinker PASSED (backend={backend})')


# ── Direct execution ──

def main() -> int:
    log('Running GRPO E2E tests directly...')
    try:
        test_grpo_twinkle()
        test_grpo_tinker()
        log('ALL GRPO TESTS PASSED')
        return 0
    except Exception as e:
        log(f'FAILED: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
