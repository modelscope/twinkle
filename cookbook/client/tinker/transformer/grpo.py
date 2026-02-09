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
from modelscope import AutoTokenizer

logger = get_logger()

# ========== Configuration ==========
BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 1024
LEARNING_RATE = 1e-5
MAX_STEPS = 10
BATCH_SIZE = 1
TEMPERATURE = 1.0
SYNC_INTERVAL = 5       # Save weights for sampler every N steps
LORA_RANK = 8


def create_countdown_dataset():
    """Create Countdown Game dataset for GRPO training."""
    logger.info("Loading Countdown dataset...")
    
    dataset = Dataset(DatasetMeta(
        "ms://zouxuhong/Countdown-Tasks-3to4", data_slice=range(500)))
    dataset.set_template(
        "Template", model_id=f'ms://{BASE_MODEL}', max_length=8192)
    dataset.map('CountdownProcessor')
    dataset.encode(add_generation_prompt=True)
    
    logger.info(f"Dataset loaded with {len(dataset)} samples")
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
    logger.info("Starting GRPO training...")
    
    # Step 1: Prepare dataset and dataloader (client-side)
    dataset = create_countdown_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True)
    
    logger.info("Dataset and tokenizer initialized")

    # Step 2: Initialize the Tinker-compatible client
    logger.info("Connecting to Tinker server...")
    service_client = init_tinker_compat_client(
        base_url='http://localhost:8000')
    
    logger.info("Creating LoRA training client...")
    # Create a LoRA training client for GRPO
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )
    
    logger.info("Training client created successfully")

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

        # ========== 6. Train the policies with GRPO loss ==========
        # Train the policies with the Advantage-Regularized policy 
        # gradient (GRPO) loss function.
        # 
        # The GRPO loss function requires:
        # 1. logprobs: The log probabilities of the tokens under the current policy
        # 2. advantages: The advantage values for each completion
        # 
        # The training data is constructed with:
        # - model_input: The full prompt + completion tokens
        # - target_tokens: The shifted tokens for next-token prediction
        # - logprobs: The log probabilities from the sampling step
        # - advantages: The computed advantage values
        training_data = []
        for i, seq in enumerate(all_sequences):
            # Build a Datum from the completion tokens with logprobs and advantages
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
            padded_advantages = [0.0] * ob_len + [advantage] * len(sampled_tokens)
            padded_logprobs = [0.0] * ob_len + logprobs
            
            # Verify lengths match
            assert len(input_tokens) == len(target_tokens) == len(padded_logprobs) == len(padded_advantages), \
                f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}, " \
                f"logprobs={len(padded_logprobs)}, advantages={len(padded_advantages)}"

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    'target_tokens': target_tokens,
                    'logprobs': types.TensorData.from_numpy(np.array(padded_logprobs, dtype=np.float32)),
                    'advantages': types.TensorData.from_numpy(np.array(padded_advantages, dtype=np.float32)),
                },
            )
            training_data.append(datum)

        if not training_data:
            logger.info(
                f"Step {step}: No training data constructed, skipping")
            step += 1
            continue

        # Forward-backward pass with importance_sampling (GRPO) loss
        # The training data already contains logprobs and advantages for the GRPO loss
        fwdbwd_future = training_client.forward_backward(
            training_data, "importance_sampling")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE))
        
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # Compute metrics from the forward-backward result
        # For importance_sampling, we get logprobs and elementwise_loss
        logprobs_list = []
        elementwise_losses = []
        for output in fwdbwd_result.loss_fn_outputs:
            if output.get('logprobs') is not None:
                logprobs_list.append(output['logprobs'].to_numpy())
            if output.get('elementwise_loss') is not None:
                elementwise_losses.append(output['elementwise_loss'].to_numpy())
        
        # Compute average loss per token (weighted by advantages)
        if elementwise_losses:
            all_losses = np.concatenate(elementwise_losses)
            avg_loss = np.mean(all_losses) if len(all_losses) > 0 else 0.0
        else:
            avg_loss = 0.0

        gc.collect()

        # ========== 7. Log ==========
        log_dict = metrics.calculate()
        log_dict['train/loss_per_token'] = float(avg_loss)
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
