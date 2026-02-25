# Tinker-Compatible Client - Math GRPO Training Example
#
# This script demonstrates Math problem training using the
# Tinker-compatible client API with save_weights_for_sampler for weight sync.
# Instead of calling sync_weights directly, it periodically saves weights and
# creates a sampling client for generation.
#
# Flow:
#   1. Prepare Math dataset (client-side)
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
from typing import List, Tuple

from twinkle_client import init_tinker_client
from twinkle import get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import Message, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import Preprocessor
from twinkle.reward.base import Reward
from twinkle.metric import CompletionRewardMetric
from twinkle.template import Template

logger = get_logger()

# ========== Configuration ==========
BASE_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
NUM_GENERATIONS = 8
MAX_NEW_TOKENS = 4096
LEARNING_RATE = 1e-4
MAX_STEPS = 1000
BATCH_SIZE = 2
TEMPERATURE = 1.0
SYNC_INTERVAL = 1  # Save weights for sampler every N steps
LORA_RANK = 8
DATA_NUM = 2000  # Number of Math samples to use

SYSTEM_PROMPT = ('You are a math assistant that values brevity. '
                 'Solve problems with minimal but correct reasoning.\n\n'
                 'Rules:\n'
                 '1. Use <step> </step> tags for reasoning\n'
                 '2. Final answer after ####\n\n'
                 'Example:\n<step>Key step1 -> Ket step 2 -> conclusion</step>\n#### 42')



class MathPreprocessor(Preprocessor):

    def __call__(self, sample):
        if sample['level'] not in ('Level 4', 'Level 5'):
            return Trajectory(messages=[], user_data=[])

        def get_boxed_answer(text):
            match = re.search(r'\\boxed{([^}]*)}', text)
            return match.group(1) if match else None

        ground_truth = get_boxed_answer(sample['solution'])
        if ground_truth is None:
            return Trajectory(messages=[], user_data=[])
        problem = sample['problem']
        return Trajectory(
            messages=[
                Message(role='system', content=SYSTEM_PROMPT),
                Message(role='user', content=problem),
            ],
            user_data=[('ground_truth', ground_truth)],
        )


# ========== Math Reward Functions ==========
class MathAccuracyReward(Reward):
    """Accuracy reward for Math: checks if the model's answer matches ground truth.

    Extracts the last '#### <number>' from model output and compares with ground truth.
    Returns 1.0 for correct, 0.0 for incorrect.
    """

    @staticmethod
    def extract_answer(completion: str) -> str:
        """Extract the last #### answer from model completion."""
        # Only check last 500 chars for efficiency
        text = completion[-500:] if len(completion) > 500 else completion
        matches = re.findall(r'####\s*([\-\d,\.\s]+)', text)
        if matches:
            return matches[-1].replace(',', '').replace(' ', '').strip()
        return ''

    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            # Get model completion (last assistant message)
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            # Get ground truth from user_data
            gt = ''
            user_data = trajectory.get('user_data', [])
            if isinstance(user_data, list):
                for item in user_data:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        if item[0] == 'ground_truth':
                            gt = str(item[1])
                            break

            predicted = self.extract_answer(completion)

            # Numeric comparison
            correct = False
            if predicted and gt:
                try:
                    correct = abs(float(predicted) - float(gt)) < 1e-5
                except (ValueError, OverflowError):
                    correct = predicted == gt

            rewards.append(1.0 if correct else 0.0)
        return rewards


class MathFormatReward(Reward):
    """Format reward: checks format and rewards shorter completions.

    Returns higher score for shorter completions (1.0 at length 100 or less).
    Returns 0.0 if format is incorrect.
    """

    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            has_think = bool(re.search(r'<step>.*?</step>', completion, re.DOTALL))
            has_answer = bool(re.search(r'####\s*[\-\d,\.]+', completion))

            if not (has_think and has_answer):
                rewards.append(0.0)
            else:
                length = len(completion)
                if length <= 100:
                    rewards.append(1.0)
                else:
                    reward = max(0.0, 1.0 - (length - 100) / 2000)
                    rewards.append(reward)

        return rewards


def create_math_dataset():
    """Create Math dataset."""
    meta = DatasetMeta(
        'ms://modelscope/competition_math',
        subset_name='default',
        split='train',
        data_slice=range(DATA_NUM),
    )
    dataset = Dataset(meta)
    dataset.set_template('Template', model_id=BASE_MODEL, max_length=4096, truncation_strategy='delete')
    dataset.map(MathPreprocessor())
    dataset.filter(lambda row: bool(row['messages']))
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(trajectories: List[Trajectory], ) -> Tuple[List[float], List[float], List[float]]:
    """Compute accuracy and format rewards for Math."""
    accuracy_reward_fn = MathAccuracyReward()
    format_reward_fn = MathFormatReward()

    accuracy_rewards = accuracy_reward_fn(trajectories, [])
    format_rewards = format_reward_fn(trajectories, [])
    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards


def main():
    logger.info('Starting Math GRPO training...')

    # Step 1: Prepare dataset and dataloader (client-side)
    dataset = create_math_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    template = Template(model_id=f'ms://{BASE_MODEL}')

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

            sampling_client = (training_client.save_weights_and_get_sampling_client(name=f'Math-step-{step}'))
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
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(trajectories)
        metrics.accumulate(
            None,
            None,
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
    save_future = training_client.save_state('Math-grpo-final')
    save_result = save_future.result()
    logger.info(f'Saved final checkpoint to {save_result.path}')


if __name__ == '__main__':
    main()
