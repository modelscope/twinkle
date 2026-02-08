# Tinker-Compatible Client - GRPO (Group Relative Policy Optimization) Training Example
#
# This script demonstrates GRPO reinforcement learning training using the
# Tinker-compatible client API with save_weights_for_sampler for weight sync.
# Instead of calling sync_weights directly, it periodically saves weights and
# creates a sampling client for generation.
#
# Flow:
#   1. Prepare Countdown dataset (client-side)
#   2. Initialize Tinker-compatible training & sampling clients
#   3. Training loop:
#      a. Every SYNC_INTERVAL steps: save_weights_for_sampler → sampling_client
#      b. Sample completions from the sampling client
#      c. Compute rewards and advantages (client-side)
#      d. Train on sampled data weighted by advantages
#      e. Optimizer step
#
# The server must be running first (see server.py and server_config.yaml).
# Requires both model and sampler services to be configured.

import gc
import numpy as np
from typing import List, Tuple

from tinker import types
from twinkle_client import init_tinker_compat_client
from twinkle import get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.server.tinker.common import input_feature_to_datum
from modelscope import AutoTokenizer

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = 'ms://Qwen/Qwen2.5-0.5B-Instruct'
NUM_GENERATIONS = 8
MAX_NEW_TOKENS = 1024
LEARNING_RATE = 1e-5
MAX_STEPS = 10
BATCH_SIZE = 4
TEMPERATURE = 1.0
SYNC_INTERVAL = 5       # Save weights for sampler every N steps
GRADIENT_ACCUMULATION_STEPS = 4


def create_countdown_dataset():
    """Create Countdown Game dataset for GRPO training."""

    dataset = Dataset(DatasetMeta(
        "ms://zouxuhong/Countdown-Tasks-3to4", data_slice=range(500)))
    dataset.set_template(
        "Template", model_id=f'ms://{BASE_MODEL}', max_length=8192)
    dataset.map('CountdownProcessor')
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[dict],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute format and accuracy rewards for Countdown game."""
    from twinkle.reward import CountDownAccuracy, FormatReward
    format_rewards = FormatReward()(trajectories, [])
    accuracy_rewards = CountDownAccuracy()(trajectories, [])
    total_rewards = [a + b for a, b in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards


def main():
    # Step 1: Prepare dataset and dataloader (client-side)
    dataset = create_countdown_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True)

    # Step 2: Initialize the Tinker-compatible client
    service_client = init_tinker_compat_client(
        base_url='http://localhost:8000')

    # Create a LoRA training client for GRPO
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    # Step 3: Setup metrics and advantage function
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = types.SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )

    # The sampling client is created on-demand via save_weights_for_sampler
    sampling_client = None

    step = 0
    for batch in dataloader:
        if step >= MAX_STEPS:
            break

        metrics.reset()
        prompts = batch if isinstance(batch, list) else [batch]

        # ========== 1. Save weights for sampler (instead of sync_weights) ==========
        if step % SYNC_INTERVAL == 0:
            logger.info(f"Step {step}: Saving weights for sampler...")
            sampling_client = (
                training_client.save_weights_and_get_sampling_client(
                    name=f'grpo-step-{step}'))
            logger.info(f"Step {step}: Sampling client ready")

        if sampling_client is None:
            logger.warning("No sampling client available, skipping step")
            step += 1
            continue

        # ========== 2. Sample completions ==========
        # Convert input features to token prompts for the sampling client
        all_sequences = []
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
            all_sequences.extend(result.sequences)

        if not all_sequences:
            logger.warning(f"Step {step}: No valid samples, skipping")
            step += 1
            continue

        # ========== 3. Build trajectories and collect logprobs ==========
        trajectories = []
        old_logps_list = []
        completion_lengths = []

        for seq in all_sequences:
            decoded_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            trajectories.append({
                'messages': [{'role': 'assistant', 'content': decoded_text}]
            })
            old_logps_list.append(
                [lp for lp in seq.logprobs] if seq.logprobs else [])
            completion_lengths.append(len(seq.tokens))

        # ========== 4. Compute rewards ==========
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(
            trajectories)
        metrics.accumulate(
            None, None,
            completion_lengths=completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            })

        # ========== 5. Compute advantages ==========
        advantages = advantage_fn(
            total_rewards,
            num_generations=NUM_GENERATIONS,
            scale='group',
        ).tolist()

        frac_zero_std = (
            1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0)
        if frac_zero_std == 1.0:
            logger.info(
                f"Step {step}: All advantages are zero, skipping training")
            step += 1
            continue

        # ========== 6. Training step ==========
        # Select samples with positive advantages for training
        # Weight them by their advantage value for GRPO-style optimization
        training_data = []
        for i, seq in enumerate(all_sequences):
            if advantages[i] <= 0:
                continue
            # Build a Datum from the completion tokens
            # Prompt tokens: weight=0 (don't compute loss on prompt)
            # Completion tokens: weight=advantage (advantage-weighted SFT)
            prompt_feature = prompts[i // NUM_GENERATIONS]
            prompt_ids = prompt_feature['input_ids']
            if hasattr(prompt_ids, 'tolist'):
                prompt_ids = prompt_ids.tolist()

            full_tokens = prompt_ids + list(seq.tokens)
            prompt_weights = [0.0] * len(prompt_ids)
            # Scale completion weights by normalized advantage
            completion_weights = [float(advantages[i])] * len(seq.tokens)

            # Shift by one for next-token prediction
            input_tokens = full_tokens[:-1]
            target_tokens = full_tokens[1:]
            weights = (prompt_weights + completion_weights)[1:]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    'target_tokens': target_tokens,
                    'weights': weights,
                },
            )
            training_data.append(datum)

        if not training_data:
            logger.info(
                f"Step {step}: No positive-advantage samples, skipping")
            step += 1
            continue

        # Forward-backward pass with cross-entropy on advantage-weighted data
        fwdbwd_future = training_client.forward_backward(
            training_data, "cross_entropy")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE))

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # Compute weighted average loss for monitoring
        logprobs = np.concatenate(
            [output['logprobs'].tolist()
             for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate(
            [d.loss_fn_inputs['weights'].tolist() for d in training_data])
        loss_per_token = -np.dot(logprobs, weights) / max(weights.sum(), 1e-8)

        gc.collect()

        # ========== 7. Log ==========
        log_dict = metrics.calculate()
        log_dict['train/loss_per_token'] = loss_per_token
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        log_dict['train/num_training_samples'] = len(training_data)
        logger.info(f"Step {step}: {log_dict}")
        step += 1

    # Save final checkpoint
    save_future = training_client.save_state("grpo-countdown-final")
    save_result = save_future.result()
    logger.info(f"Saved final checkpoint to {save_result.path}")


if __name__ == '__main__':
    main()
