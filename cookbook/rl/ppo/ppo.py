"""Standard PPO training on GSM8K with a LoRA policy and full-parameter critic.

The first implementation supports the Transformers/Accelerate-FSDP backend. Policy,
critic, and vLLM sampler use separate GPU groups. The frozen policy base model is
used as the reference policy.
"""
import random
from typing import Any, Dict, List, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GAEAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric, PPOMetric, PPOValueMetric
from twinkle.model import TransformersModel, TransformersValueModel
from twinkle.processor import InputProcessor
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
from twinkle.sampler import vLLMSampler

logger = get_logger()
args = CLI.from_args()

MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'
POLICY_GPUS = args.infra.model_gpus or 4
CRITIC_GPUS = args.infra.critic_model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = POLICY_GPUS + CRITIC_GPUS + SAMPLER_GPUS
NUM_GENERATIONS = args.rl.num_generations or 4
MAX_NEW_TOKENS = args.sampling.max_tokens or 1024
POLICY_LR = args.optimizer.learning_rate or 1e-5
CRITIC_LR = args.rl.critic_learning_rate
MAX_STEPS = args.training.max_steps or 200
BATCH_SIZE = args.training.batch_size or 4
MINI_BATCH_SIZE = args.training.mini_batch_size or 4
MICRO_BATCH_SIZE = args.training.micro_batch_size or 1
# Number of policy/value updates over each rollout batch.  Reuse the common
# training argument whaohile preserving PPO's historical default.
PPO_EPOCHS = args.training.num_train_epochs if args.training.num_train_epochs is not None else 4
SAVE_STEPS = args.training.save_steps or 50
ADAPTER_NAME = args.lora.adapter_name or 'default'


def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=400)
    dataset.map(GSM8KProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(trajectories: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float]]:
    accuracy = GSM8KAccuracyReward()(trajectories)
    formatting = GSM8KFormatReward()(trajectories)
    return [a + f for a, f in zip(accuracy, formatting)], formatting, accuracy


def response_rows(full_values, trajectories) -> List[List[float]]:
    """Extract response-token rows from collected model outputs."""
    import torch

    value_rows = []
    tensors = full_values if isinstance(full_values, list) else [full_values]
    for tensor in tensors:
        if tensor is None:
            continue
        tensor = torch.as_tensor(tensor)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        value_rows.extend(tensor)
    if len(value_rows) != len(trajectories):
        raise ValueError(f'model output batch mismatch: {len(value_rows)} rows for {len(trajectories)} trajectories')

    rows = []
    for value_row, trajectory in zip(value_rows, trajectories):
        mask = torch.as_tensor(trajectory['labels'], device=value_row.device) != -100
        rows.append(value_row[:mask.numel()][mask].detach().float().cpu().tolist())
    return rows


def main():
    critic_start = POLICY_GPUS
    sampler_start = POLICY_GPUS + CRITIC_GPUS
    groups = [
        DeviceGroup(name='policy', ranks=list(range(POLICY_GPUS)), device_type='GPU'),
        DeviceGroup(name='critic', ranks=list(range(critic_start, sampler_start)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(sampler_start, NUM_GPUS)), device_type='GPU'),
    ]
    policy_mesh = DeviceMesh.from_sizes(world_size=POLICY_GPUS, fsdp_size=POLICY_GPUS)
    critic_mesh = DeviceMesh.from_sizes(world_size=CRITIC_GPUS, fsdp_size=CRITIC_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=groups, lazy_collect=False)

    policy = TransformersModel(
        model_id=MODEL_ID, device_mesh=policy_mesh, remote_group='policy')
    lora_config = LoraConfig(
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
    )
    policy.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    policy.set_optimizer('AdamW', lr=POLICY_LR)
    policy.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    policy.set_loss('PPOLoss', epsilon=args.loss.epsilon, entropy_coef=args.loss.entropy_coef)
    policy.add_metric(PPOMetric, epsilon=args.loss.epsilon)
    policy.set_processor(InputProcessor)
    policy.set_template('Qwen3_5Template', model_id=MODEL_ID)

    critic = TransformersValueModel(
        model_id=MODEL_ID, device_mesh=critic_mesh, remote_group='critic')
    critic.set_optimizer('AdamW', lr=CRITIC_LR)
    critic.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    critic.set_loss('PPOValueLoss', epsilon=args.loss.value_clip)
    critic.add_metric(PPOValueMetric, epsilon=args.loss.value_clip)
    critic.set_processor(InputProcessor)
    critic.set_template('Qwen3_5Template', model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 400 + MAX_NEW_TOKENS,
            'max_lora_rank': 32,
            'enable_lora': True,
            'tensor_parallel_size': 1,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)
    checkpoint_manager = CheckpointEngineManager(model=policy, sampler=sampler)
    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=policy_mesh,
        remote_group='policy',
    )
    gae = GAEAdvantage(args.rl.gamma, args.rl.gae_lambda, args.rl.normalize_advantages)
    reward_metric = CompletionRewardMetric()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1)

    optim_step = 0
    rollout_step = 0
    logger.info(get_device_placement())
    while optim_step < MAX_STEPS:
        for batch in dataloader:
            if optim_step >= MAX_STEPS:
                break
            reward_metric.reset()
            prompts = batch if isinstance(batch, list) else [batch]
            checkpoint_manager.sync_weights(merge_and_sync=False)
            sampler.reset_prefix_cache()
            expanded = [prompt for prompt in prompts for _ in range(NUM_GENERATIONS)]
            samples = sampler.sample(expanded, sampling_params)

            trajectories, old_logps, lengths = [], [], []
            for response in samples:
                for sequence in response.sequences:
                    trajectories.append(sequence.new_input_feature)
                    old_logps.append([entry[0][1] for entry in sequence.logprobs])
                    lengths.append(len(sequence.tokens))
            rewards, format_rewards, accuracy_rewards = compute_rewards(trajectories)
            reward_metric.accumulate(
                completion_lengths=lengths,
                rewards={'total': rewards, 'format': format_rewards, 'accuracy': accuracy_rewards},
            )

            reference = policy.forward_only(inputs=trajectories, disable_lora=True)
            ref_logps = response_rows(reference['logps'], trajectories)
            critic_outputs = critic.forward_only(inputs=trajectories)
            old_values = response_rows(critic_outputs['values'], trajectories)
            token_rewards = gae.build_token_rewards(
                rewards, lengths, old_logps=old_logps, ref_logps=ref_logps, kl_coef=args.rl.kl_coef)
            max_len = max(lengths)
            padded_rewards = [row + [0.0] * (max_len - len(row)) for row in token_rewards]
            padded_values = [row + [0.0] * (max_len - len(row)) for row in old_values]
            masks = [[True] * length + [False] * (max_len - length) for length in lengths]
            advantages, returns = gae(padded_rewards, padded_values, masks=masks)
            advantages = [advantages[i, :length].tolist() for i, length in enumerate(lengths)]
            returns = [returns[i, :length].tolist() for i, length in enumerate(lengths)]

            indices = list(range(len(trajectories)))
            for _ in range(PPO_EPOCHS):
                random.shuffle(indices)
                for start in range(0, len(indices), MINI_BATCH_SIZE):
                    chosen = indices[start:start + MINI_BATCH_SIZE]
                    mb_inputs = [trajectories[i] for i in chosen]
                    mb_old_logps = [old_logps[i] for i in chosen]
                    mb_old_values = [old_values[i] for i in chosen]
                    mb_advantages = [advantages[i] for i in chosen]
                    mb_returns = [returns[i] for i in chosen]
                    policy.forward_backward(
                        inputs=mb_inputs,
                        old_logps=mb_old_logps,
                        advantages=mb_advantages,
                        micro_batch_size=MICRO_BATCH_SIZE,
                    )
                    policy.clip_grad_and_step()
                    critic.forward_backward(
                        inputs=mb_inputs,
                        old_values=mb_old_values,
                        returns=mb_returns,
                        advantages=mb_advantages,
                        micro_batch_size=MICRO_BATCH_SIZE,
                    )
                    critic.clip_grad_and_step()
                    optim_step += 1
                    if optim_step % SAVE_STEPS == 0:
                        policy.save(f'ppo-policy-checkpoint-{optim_step}')
                        critic.save(f'ppo-critic-checkpoint-{optim_step}')
                    if optim_step >= MAX_STEPS:
                        break
                if optim_step >= MAX_STEPS:
                    break

            logs = reward_metric.calculate()
            logs.update(policy.calculate_metric(is_training=True))
            logs.update(critic.calculate_metric(is_training=True))
            rollout_step += 1
            logger.info(f'[Rollout {rollout_step}, optim step {optim_step}/{MAX_STEPS}] {logs}')

    policy.save('ppo-policy-final')
    critic.save('ppo-critic-final')


if __name__ == '__main__':
    main()
