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
import re
import numpy as np
from typing import List, Tuple

from tinker import types
from twinkle_client import init_tinker_compat_client
from twinkle import get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.dataloader import DataLoader
from twinkle.preprocessor import Preprocessor
from twinkle.reward.base import Reward
from twinkle.data_format import Trajectory, InputFeature, Message
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from modelscope import AutoTokenizer

logger = get_logger()

# ========== Configuration ==========
BASE_MODEL = 'Qwen/Qwen2.5-3B-Instruct'
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 2048
LEARNING_RATE = 1e-5
MAX_STEPS = 100
BATCH_SIZE = 2
TEMPERATURE = 1.0
SYNC_INTERVAL = 1       # Save weights for sampler every N steps
LORA_RANK = 8
DATA_NUM = 1000         # Number of GSM8K samples to use

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "Show your reasoning in <think> </think> tags, then give the final "
    "numerical answer after ####.\n"
    "For example:\n<think> ... reasoning ... </think>\n#### 42"
)


class GSM8KProcessor(Preprocessor):
    """Preprocessor for GSM8K dataset.

    GSM8K fields: question (str), answer (str ending with '#### <number>')
    Extracts the ground truth number and stores it in user_data for reward.
    """

    @staticmethod
    def extract_ground_truth(answer_str: str) -> str:
        """Extract the number after '####' from GSM8K answer."""
        match = re.search(r'####\s*([\-\d,\.]+)', answer_str)
        if match:
            return match.group(1).replace(',', '').strip()
        return ''

    def __call__(self, row) -> Trajectory:
        question = row['question']
        answer = row.get('answer', '')
        ground_truth = self.extract_ground_truth(answer)

        messages = [
            Message(role='system', content=SYSTEM_PROMPT),
            Message(role='user', content=question),
        ]
        return Trajectory(
            messages=messages,
            user_data=[('ground_truth', ground_truth)],
        )


# ========== GSM8K Reward Functions ==========
class GSM8KAccuracyReward(Reward):
    """Accuracy reward for GSM8K: checks if the model's answer matches ground truth.

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

    def __call__(
        self, trajectories: List[Trajectory], ground_truths: List[Trajectory]
    ) -> List[float]:
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


class GSM8KFormatReward(Reward):
    """Format reward: checks if output contains <think>...</think> tag.

    Returns 1.0 if format is correct, 0.0 otherwise.
    """

    def __call__(
        self, trajectories: List[Trajectory], ground_truths: List[Trajectory]
    ) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_think = bool(
                re.search(r'<think>.*?</think>', completion, re.DOTALL)
            )
            has_answer = bool(re.search(r'####\s*[\-\d,\.]+', completion))
            rewards.append(1.0 if (has_think and has_answer) else 0.0)
        return rewards


def create_gsm8k_dataset():
    """Create GSM8K dataset."""
    meta = DatasetMeta(
        "ms://modelscope/gsm8k",
        subset_name='main', split='train',
        data_slice=range(DATA_NUM),
    )
    dataset = Dataset(meta)
    dataset.set_template("Template", model_id=f'ms://{BASE_MODEL}', max_length=2048)
    dataset.map(GSM8KProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Trajectory],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute accuracy and format rewards for GSM8K."""
    accuracy_reward_fn = GSM8KAccuracyReward()
    format_reward_fn = GSM8KFormatReward()

    accuracy_rewards = accuracy_reward_fn(trajectories, [])
    format_rewards = format_reward_fn(trajectories, [])
    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards


def main():
    logger.info("Starting GSM8K GRPO training...")
    
    # Step 1: Prepare dataset and dataloader (client-side)
    dataset = create_gsm8k_dataset()
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
                    name=f'gsm8k-step-{step}'))
            logger.info(f"Step {step}: Sampling client ready")

        if sampling_client is None:
            logger.warning("No sampling client available, skipping step")
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
            logger.warning(f"Step {step}: No valid samples, skipping")
            step += 1
            continue

        # ========== 3. Build trajectories and collect logprobs ==========
        trajectories = []
        old_logps_list = []
        completion_lengths = []

        for idx, seq in enumerate(all_sequences):
            decoded_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            # Use the corresponding user data for this sequence
            trajectories.append({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': 'Math problem'},  # Placeholder
                    {'role': 'assistant', 'content': decoded_text}
                ],
                'user_data': all_user_data[idx]
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
    save_future = training_client.save_state("gsm8k-grpo-final")
    save_result = save_future.result()
    logger.info(f"Saved final checkpoint to {save_result.path}")


if __name__ == '__main__':
    main()