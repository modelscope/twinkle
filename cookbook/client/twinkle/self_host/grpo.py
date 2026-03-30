# Twinkle Client - GRPO (Group Relative Policy Optimization) Training Example
#
# This script demonstrates GRPO reinforcement learning training using the
# Twinkle client API with model.save() + adapter_uri for weight sync.
# Instead of calling sync_weights directly, it periodically saves model weights
# and passes the checkpoint path to the sampler as adapter_uri.
#
# Flow:
#   1. Prepare Countdown dataset (client-side)
#   2. Initialize Twinkle client, model, and sampler
#   3. Configure model with GRPOLoss, optimizer, LR scheduler
#   4. Training loop:
#      a. Every SYNC_INTERVAL steps: model.save() → get twinkle_path
#      b. sampler.sample(inputs, adapter_uri=twinkle_path, num_samples=N)
#      c. Compute rewards and advantages (client-side)
#      d. model.forward_backward(inputs, advantages, old_logps)
#      e. Optimizer step
#
# The server must be running first (see server.py and server_config.yaml).
# Requires both model and sampler services to be configured.

import dotenv

dotenv.load_dotenv('.env')

import gc
import os
from peft import LoraConfig
from typing import List, Tuple, Dict, Any

from twinkle import get_logger
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
from twinkle.advantage import GRPOAdvantage
from twinkle.dataset import DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle_client import init_twinkle_client
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 1024
LEARNING_RATE = 1e-5
MAX_STEPS = 10
BATCH_SIZE = 2
TEMPERATURE = 1.0
SYNC_INTERVAL = 1  # Save weights for sampler every N steps
GRADIENT_ACCUMULATION_STEPS = 4


def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=2048)
    dataset.map('GSM8KProcessor')
    dataset.encode(add_generation_prompt=True)
    return dataset

def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_reward_fn = GSM8KAccuracyReward()
    format_reward_fn = GSM8KFormatReward()

    accuracy_rewards = accuracy_reward_fn(trajectories)
    format_rewards = format_reward_fn(trajectories)
    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards


def train():
    # Step 1: Initialize the Twinkle client
    client = init_twinkle_client(
        base_url='http://127.0.0.1:8000',
        api_key=os.environ.get('MODELSCOPE_TOKEN'),
    )

    # Step 2: Prepare dataset and dataloader
    dataset = create_gsm8k_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    # Step 3: Configure the training model
    model = MultiLoraTransformersModel(model_id=MODEL_ID)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model.add_adapter_to_model(
        'default',
        lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    # Set GRPO loss (the key difference from SFT training)
    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)

    # Set optimizer and LR scheduler
    model.set_optimizer('Adam', lr=LEARNING_RATE)
    # Set LR scheduler (if server use megatron, don't support set self.lr_scheduler)
    # model.set_lr_scheduler(
    #     'CosineWarmupScheduler',
    #     num_warmup_steps=500,
    #     num_training_steps=MAX_STEPS,
    # )

    # Set processor and template for encoding inputs
    model.set_processor('InputProcessor')
    model.set_template('Template', model_id=MODEL_ID)

    # Step 4: Configure the sampler
    sampler = vLLMSampler(model_id=MODEL_ID)
    sampler.set_template('Template', model_id=MODEL_ID)

    # Step 5: Setup metrics and advantage function
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = {
        'max_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'top_p': 0.95,
        'num_samples': NUM_GENERATIONS,
        'logprobs': 1,
    }

    # Track the current adapter path for sampling
    current_adapter_uri = None

    step = 0
    for batch in dataloader:
        if step >= MAX_STEPS:
            break

        metrics.reset()
        prompts = batch if isinstance(batch, list) else [batch]

        # ========== 1. Save weights and update adapter_uri ==========
        # Instead of sync_weights, save the model checkpoint and pass
        # the resulting path to the sampler as adapter_uri
        if step % SYNC_INTERVAL == 0:
            logger.info(f'Step {step}: Saving weights for sampler...')
            result = model.save(
                name=f'grpo-sampler-step-{step}',
                save_optimizer=False,
            )
            current_adapter_uri = result.twinkle_path
            logger.info(f'Step {step}: Saved weights to {current_adapter_uri}')

        # ========== 2. Sample completions ==========
        sample_responses = sampler.sample(
            inputs=prompts,
            sampling_params=sampling_params,
            adapter_uri=current_adapter_uri,
        )

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        # ========== 3. Compute rewards ==========

        total_rewards, format_rewards, accuracy_rewards = compute_rewards(
            all_input_data
        )
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            },
        )


        # ========== 4. Compute advantages ==========
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

        # ========== 5. Training step (GRPO) ==========
        # forward_backward with GRPO loss: passes advantages and old_logps
        # to the server-side GRPOLoss for proper policy optimization
        model.forward_backward(
            inputs=all_input_data,
            advantages=advantages,
            old_logps=all_old_logps,
        )

        # Gradient clipping and optimizer step
        model.clip_grad_and_step()

        gc.collect()

        # ========== 6. Log ==========
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True).result)
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        logger.info(f'Step {step}: {log_dict}')
        step += 1
        metrics.reset()

    # Save final checkpoint
    result = model.save(name='grpo-countdown-final', save_optimizer=True)
    logger.info(f'Saved final checkpoint: {result}')


if __name__ == '__main__':
    train()
