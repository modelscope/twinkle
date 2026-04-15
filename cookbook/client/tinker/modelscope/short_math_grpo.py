# Tinker-Compatible Client - GSM8K GRPO Training Example
#
# This script demonstrates GSM8K math problem training using the
# Tinker-compatible client API with save_weights_for_sampler for weight sync.
# Instead of calling sync_weights directly, it periodically saves weights and
# creates a sampling client for generation.
#
# Flow:
#   1. Prepare GSM8K dataset (client-side)
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
import os
import re
from tinker import types
from typing import List, Tuple, Dict, Any

from twinkle import init_tinker_client
from twinkle import get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.metric import CompletionRewardMetric
from twinkle.template import Qwen3_5Template

logger = get_logger()

# ========== Configuration ==========
BASE_MODEL = 'Qwen/Qwen3.6-35B-A3B'
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 4096
LEARNING_RATE = 2e-5
MAX_STEPS = 1000
BATCH_SIZE = 2
TEMPERATURE = 1.0
SYNC_INTERVAL = 1  # Save weights for sampler every N steps
LORA_RANK = 16
DATA_NUM = 2000  # Number of Math samples to use

SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                 'and put your final answer within \\boxed{}.')


# ========== Reward Functions ==========
class GSM8KBrevityReward(Reward):
    """Brevity reward: rewards shorter completions that contain a valid answer.

    Returns 0.0 if no valid answer format (\\boxed{} or ####).
    Otherwise returns higher score for shorter completions (1.0 at <=200 chars).
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
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
                rewards.append(0.0)
            else:
                length = len(completion)
                if length <= 200:
                    rewards.append(1.0)
                else:
                    rewards.append(max(0.0, 1.0 - (length - 200) / 3000))
        return rewards


# ========== Dataset ==========
def create_gsm8k_dataset():
    """Create GSM8K dataset."""
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train', data_slice=range(DATA_NUM)))
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{BASE_MODEL}', max_length=4096,
                         truncation_strategy='delete', enable_thinking=True)
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute accuracy and brevity rewards for GSM8K."""
    accuracy_reward_fn = GSM8KAccuracyReward()
    brevity_reward_fn = GSM8KBrevityReward()

    accuracy_rewards = accuracy_reward_fn(trajectories)
    brevity_rewards = brevity_reward_fn(trajectories)
    total_rewards = [a + b for a, b in zip(accuracy_rewards, brevity_rewards)]
    return total_rewards, brevity_rewards, accuracy_rewards


def main():
    logger.info('Starting GSM8K GRPO training...')

    # Step 1: Prepare dataset and dataloader (client-side)
    dataset = create_gsm8k_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0)
    template = Qwen3_5Template(model_id=f'ms://{BASE_MODEL}')

    logger.info('Dataset and template initialized')

    # Step 2: Initialize the Tinker-compatible client
    logger.info('Connecting to Tinker server...')
    init_tinker_client()

    from tinker import ServiceClient
    service_client = ServiceClient(
        base_url='http://www.modelscope.cn/twinkle',
        api_key=os.environ.get('MODELSCOPE_TOKEN')
    )

    logger.info('Creating LoRA training client...')
    # Create a LoRA training client for GRPO
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    logger.info('Training client created successfully')

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
            logger.info(f'Step {step}: Saving weights for sampler...')

            sampling_client = (training_client.save_weights_and_get_sampling_client(name=f'GSM8K-step-{step}'))
            logger.info(f'Step {step}: Sampling client ready')

        if sampling_client is None:
            logger.warning('No sampling client available, skipping step')
            step += 1
            continue

        # ========== 2. Sample completions ==========
        # Convert input features to token prompts for the sampling client
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
            # Store both sequences and user data
            for _ in range(NUM_GENERATIONS):
                all_user_data.append(prompt_feature.get('user_data', []))
            all_sequences.extend(result.sequences)

        if not all_sequences:
            logger.warning(f'Step {step}: No valid samples, skipping')
            step += 1
            continue

        # ========== 3. Build trajectories and collect logprobs ==========
        trajectories = []
        old_logps_list = []
        completion_lengths = []

        for idx, seq in enumerate(all_sequences):
            decoded_text = template.decode(seq.tokens, skip_special_tokens=True)
            # Use the corresponding user data for this sequence
            trajectories.append({
                'messages': [
                    {
                        'role': 'system',
                        'content': SYSTEM_PROMPT
                    },
                    {
                        'role': 'user',
                        'content': 'Math problem'
                    },  # Placeholder
                    {
                        'role': 'assistant',
                        'content': decoded_text
                    }
                ],
                'user_data':
                all_user_data[idx]
            })
            old_logps_list.append([lp for lp in seq.logprobs] if seq.logprobs else [])
            completion_lengths.append(len(seq.tokens))

        # ========== 4. Compute rewards ==========
        total_rewards, brevity_rewards, accuracy_rewards = compute_rewards(trajectories)
        metrics.accumulate(
            completion_lengths=completion_lengths,
            rewards={
                'total': total_rewards,
                'brevity': brevity_rewards,
                'accuracy': accuracy_rewards,
            })

        # ========== 5. Compute advantages ==========
        advantages = advantage_fn(
            total_rewards,
            num_generations=NUM_GENERATIONS,
            scale='group',
        ).tolist()

        frac_zero_std = (1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0)
        if frac_zero_std == 1.0:
            logger.info(f'Step {step}: All advantages are zero, skipping training')
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
            weights = [0] * ob_len + [1] * len(sampled_tokens)
            padded_advantages = [0.0] * ob_len + [advantage] * len(sampled_tokens)
            padded_logprobs = [0.0] * ob_len + logprobs

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
            logger.info(f'Step {step}: No training data constructed, skipping')
            step += 1
            continue

        # Forward-backward pass with importance_sampling (GRPO) loss
        # The training data already contains logprobs and advantages for the GRPO loss
        fwdbwd_result = training_client.forward_backward(training_data, 'importance_sampling').result()

        optim_result = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()

        gc.collect()

        # ========== 7. Log ==========
        log_dict = metrics.calculate()
        if optim_result.metrics:
            log_dict.update(optim_result.metrics)
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        log_dict['train/num_training_samples'] = len(training_data)
        logger.info(f'Step {step}: {log_dict}')
        step += 1

    # Save final checkpoint
    save_future = training_client.save_state('gsm8k-grpo-final')
    save_result = save_future.result()
    logger.info(f'Saved final checkpoint to {save_result.path}')


if __name__ == '__main__':
    main()
