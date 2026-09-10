"""Synchronous algorithm-correctness baseline for SAO on GSM8K.

This intentionally retains a rollout-batch barrier. It validates single rollout,
DIS, fixed critic targets, frozen-attention critic training, and a 2:1
critic-to-actor update ratio. It is not the asynchronous SAO pipeline.
"""
import random
from typing import Any, Dict, List, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GAEAdvantage, SAOGAEAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric, SAOMetric
from twinkle.model import TransformersModel, TransformersValueModel
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
from twinkle.sampler import vLLMSampler

logger = get_logger()
args = CLI.from_args()

MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'
POLICY_GPUS = args.infra.model_gpus or 4
CRITIC_GPUS = args.infra.critic_model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = POLICY_GPUS + CRITIC_GPUS + SAMPLER_GPUS
MAX_NEW_TOKENS = args.sampling.max_tokens or 1024
POLICY_LR = args.optimizer.learning_rate
CRITIC_LR = args.rl.critic_learning_rate
MAX_STEPS = args.training.max_steps or 200
BATCH_SIZE = args.training.batch_size or 4
MINI_BATCH_SIZE = args.training.mini_batch_size or 4
MICRO_BATCH_SIZE = args.training.micro_batch_size or 1
SAVE_STEPS = args.training.save_steps or 50
ADAPTER_NAME = args.lora.adapter_name or 'default'
CRITIC_UPDATES = args.rl.critic_updates_per_actor_update
EPSILON_HIGH = 5.0 if args.loss.epsilon_high is None else args.loss.epsilon_high


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
        if value_row.numel() < mask.numel():
            raise ValueError(
                f'value sequence is shorter than labels: values={value_row.numel()}, labels={mask.numel()}')
        response_values = value_row[:mask.numel()][mask]
        if response_values.numel() == 0:
            raise ValueError('trajectory contains no action-token value predictions')
        rows.append(response_values.detach().float().cpu().tolist())
    return rows


def pad_rows(rows, lengths, fill=0.0):
    max_len = max(lengths)
    return [list(row) + [fill] * (max_len - len(row)) for row in rows]


def main():
    if args.rl.num_generations != 1:
        raise ValueError('SAO requires --num-generations 1')
    if CRITIC_UPDATES <= 0:
        raise ValueError('--critic-updates-per-actor-update must be positive')
    if min(BATCH_SIZE, MINI_BATCH_SIZE, MICRO_BATCH_SIZE) <= 0:
        raise ValueError('batch-size, mini-batch-size, and micro-batch-size must be positive')
    if MINI_BATCH_SIZE > BATCH_SIZE:
        raise ValueError('--mini-batch-size cannot exceed --batch-size')
    if MINI_BATCH_SIZE % MICRO_BATCH_SIZE != 0:
        raise ValueError('--mini-batch-size must be divisible by --micro-batch-size')
    # DIS compares vLLM rollout log-probs with the learner's raw model log-probs.
    # Temperature/top-k/top-p/repetition transforms would make those two
    # distributions different and invalidate the importance ratio.
    if args.sampling.temperature != 1.0:
        raise ValueError('SAO requires --temperature 1.0 for an exact rollout/current log-prob ratio')
    if args.sampling.top_p != 1.0 or args.sampling.top_k != -1:
        raise ValueError('SAO requires --top-p 1.0 and --top-k -1 for an exact behavior-policy log-prob')
    if args.sampling.repetition_penalty != 1.0:
        raise ValueError('SAO requires --repetition-penalty 1.0 for an exact behavior-policy log-prob')
    if BATCH_SIZE % MINI_BATCH_SIZE != 0:
        logger.warning('batch-size is not divisible by mini-batch-size; the final actor/critic update '
                       'of each rollout batch will use a smaller mini-batch')
    actor_updates_per_rollout = (BATCH_SIZE + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE
    if actor_updates_per_rollout != 1:
        logger.warning(
            f'Each rollout batch performs {actor_updates_per_rollout} actor optimizer steps. '
            'For the paper-style one update over a global batch of 128, set '
            '--batch-size 128 --mini-batch-size 128.')

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

    policy = TransformersModel(model_id=MODEL_ID, device_mesh=policy_mesh, remote_group='policy')
    policy.add_adapter_to_model(
        ADAPTER_NAME,
        LoraConfig(
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            r=32, lora_alpha=64, lora_dropout=0.05),
        gradient_accumulation_steps=1,
    )
    policy.set_optimizer('AdamW', lr=POLICY_LR)
    policy.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    policy.set_loss(
        'SAOLoss', epsilon_low=args.loss.epsilon_low, epsilon_high=EPSILON_HIGH,
        detach_importance_weight=args.loss.detach_importance_weight, entropy_coef=args.loss.entropy_coef)
    policy.add_metric(SAOMetric, epsilon=args.loss.epsilon_low, epsilon_high=EPSILON_HIGH)
    policy.set_processor(InputProcessor)
    policy.set_template('Qwen3_5Template', model_id=MODEL_ID)

    critic = TransformersValueModel(model_id=MODEL_ID, device_mesh=critic_mesh, remote_group='critic')
    if args.rl.freeze_critic_attention:
        frozen = critic.freeze_attention_for_value_training()
        logger.info(f'Frozen-attention critic: {frozen}')
    logger.info(f'Critic parameters: {critic.trainable_parameter_summary()}')
    critic.set_optimizer('AdamW', lr=CRITIC_LR)
    critic.set_lr_scheduler(
        'LinearWarmupScheduler', num_warmup_steps=10,
        num_training_steps=MAX_STEPS * CRITIC_UPDATES)
    critic.set_loss('SAOValueLoss')
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
        dataset=create_gsm8k_dataset, batch_size=BATCH_SIZE, min_batch_size=BATCH_SIZE,
        device_mesh=policy_mesh, remote_group='policy')
    critic_gae = SAOGAEAdvantage(
        gamma=args.rl.gamma, gae_lambda=args.rl.sao_critic_lambda, normalize=False)
    actor_gae = SAOGAEAdvantage(
        gamma=args.rl.gamma, alpha=args.rl.sao_alpha,
        gae_lambda=None if args.rl.sao_policy_lambda_adaptive else args.rl.gae_lambda,
        normalize=args.rl.normalize_advantages)
    reward_metric = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        seed=args.sampling.seed,
        stop=args.sampling.stop,
        temperature=args.sampling.temperature,
        top_k=args.sampling.top_k,
        top_p=args.sampling.top_p,
        repetition_penalty=args.sampling.repetition_penalty,
        num_samples=1,
        logprobs=1,
    )

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
            samples = sampler.sample(prompts, sampling_params)

            trajectories, rollout_logps, lengths = [], [], []
            for response in samples:
                for sequence in response.sequences:
                    if not sequence.tokens or sequence.logprobs is None:
                        raise ValueError('SAO rollout must contain generated tokens and log-probabilities')
                    action_count = sum(label != -100 for label in sequence.new_input_feature['labels'])
                    if len(sequence.tokens) != len(sequence.logprobs) or action_count != len(sequence.tokens):
                        raise ValueError(
                            'SAO rollout token alignment failed: '
                            f'tokens={len(sequence.tokens)}, logprobs={len(sequence.logprobs)}, '
                            f'action_labels={action_count}')
                    trajectories.append(sequence.new_input_feature)
                    rollout_logps.append([entry[0][1] for entry in sequence.logprobs])
                    lengths.append(len(sequence.tokens))
            if not trajectories:
                logger.warning('No trajectories in rollout batch; skipping learner update')
                continue
            rewards, format_rewards, accuracy_rewards = compute_rewards(trajectories)
            reward_metric.accumulate(
                completion_lengths=lengths,
                rewards={'total': rewards, 'format': format_rewards, 'accuracy': accuracy_rewards})
            token_rewards = GAEAdvantage.build_token_rewards(rewards, lengths)
            padded_rewards = pad_rows(token_rewards, lengths)
            masks = [[True] * length + [False] * (max(lengths) - length) for length in lengths]
            terminated = [True] * len(trajectories)
            truncated = [False] * len(trajectories)

            initial = critic.forward_only(inputs=trajectories)
            initial_values = response_rows(initial['values'], trajectories)
            _, fixed_returns = critic_gae(
                padded_rewards, pad_rows(initial_values, lengths), action_masks=masks,
                terminated=terminated, truncated=truncated, effective_lengths=lengths)
            fixed_returns = [fixed_returns[i, :length].tolist() for i, length in enumerate(lengths)]

            indices = list(range(len(trajectories)))
            random.shuffle(indices)
            for start in range(0, len(indices), MINI_BATCH_SIZE):
                chosen = indices[start:start + MINI_BATCH_SIZE]
                mb_inputs = [trajectories[i] for i in chosen]
                mb_returns = [fixed_returns[i] for i in chosen]
                for _ in range(CRITIC_UPDATES):
                    critic.forward_backward(
                        inputs=mb_inputs, returns=mb_returns, micro_batch_size=MICRO_BATCH_SIZE)
                    critic.clip_grad_and_step()

                updated = critic.forward_only(inputs=mb_inputs)
                new_values = response_rows(updated['values'], mb_inputs)
                mb_lengths = [lengths[i] for i in chosen]
                mb_rewards = [token_rewards[i] for i in chosen]
                mb_masks = [[True] * length + [False] * (max(mb_lengths) - length) for length in mb_lengths]
                advantages, _ = actor_gae(
                    pad_rows(mb_rewards, mb_lengths), pad_rows(new_values, mb_lengths), action_masks=mb_masks,
                    terminated=[True] * len(chosen), truncated=[False] * len(chosen),
                    effective_lengths=mb_lengths)
                mb_advantages = [advantages[i, :length].tolist() for i, length in enumerate(mb_lengths)]
                policy.forward_backward(
                    inputs=mb_inputs, old_logps=[rollout_logps[i] for i in chosen], advantages=mb_advantages,
                    micro_batch_size=MICRO_BATCH_SIZE)
                policy.clip_grad_and_step()
                optim_step += 1
                if optim_step % SAVE_STEPS == 0:
                    policy.save(f'sao-policy-checkpoint-{optim_step}')
                    critic.save(f'sao-critic-checkpoint-{optim_step}')
                if optim_step >= MAX_STEPS:
                    break

            logs = reward_metric.calculate()
            logs.update(policy.calculate_metric(is_training=True))
            logs.update({f'critic/{key}': value for key, value in critic.calculate_metric(is_training=True).items()})
            logs['train/critic_updates_per_actor_update'] = CRITIC_UPDATES
            logs['train/actor_updates_per_rollout_batch'] = actor_updates_per_rollout
            rollout_step += 1
            logger.info(f'[SAO sync rollout {rollout_step}, actor step {optim_step}/{MAX_STEPS}] {logs}')

    policy.save('sao-policy-final')
    critic.save('sao-critic-final')


if __name__ == '__main__':
    main()
