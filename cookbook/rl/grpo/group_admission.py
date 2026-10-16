# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end GSM8K GRPO training with budget-aware group admission.

This example follows ``short_math_grpo.py`` and inserts complete-group
admission plus bounded replacement sampling between rollout and GRPO.  Weight
synchronization happens once per batch, so initial and replacement groups use
the same rollout policy snapshot.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.advantage.group_admission import (
    GroupAdmissionConfig,
    GroupAdmissionDecision,
    GroupAdmissionPolicy,
    SamplingBudgetConfig,
    SamplingBudgetController,
    SamplingBudgetState,
    group_admission_metrics,
)
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'
USE_MEGATRON = args.model.strategy != 'native_fsdp'

MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = args.rl.num_generations or 8
MAX_NEW_TOKENS = args.sampling.max_tokens or 4096
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
MAX_STEPS = args.training.max_steps or 1000
BATCH_SIZE = args.training.batch_size or 8
MINI_BATCH_SIZE = args.training.mini_batch_size or 8
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
ADAPTER_NAME = args.lora.adapter_name or 'default'
SAVE_STEPS = args.training.save_steps or 1000
LORA_RANK = args.lora.lora_r or 16

if MINI_BATCH_SIZE % NUM_GENERATIONS != 0:
    raise ValueError(
        'mini_batch_size must be divisible by num_generations to preserve complete group boundaries, '
        f'but got {MINI_BATCH_SIZE} % {NUM_GENERATIONS} != 0')

# The defaults permit at most one extra complete group per target group.  The
# environment variables make the cost envelope adjustable without coupling it
# to a particular sampler implementation.
MAX_EXTRA_GROUPS = int(os.getenv('GROUP_ADMISSION_MAX_EXTRA_GROUPS', BATCH_SIZE))
MAX_RESAMPLE_ROUNDS = int(os.getenv('GROUP_ADMISSION_MAX_RESAMPLE_ROUNDS', '2'))

SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                 'and put your final answer within \\boxed{}.')


# ========== Reward Functions ==========
class GSM8KBrevityReward(Reward):
    """Reward short completions that contain a recognizable final answer."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for message in reversed(messages):
                if message.get('role') == 'assistant':
                    completion = message.get('content', '')
                    break

            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion))
            if not has_answer:
                rewards.append(0.0)
            elif len(completion) <= 300:
                rewards.append(1.0)
            else:
                rewards.append(max(0.0, 1.0 - (len(completion) - 300) / 3000))
        return rewards


# ========== Dataset ==========
def create_gsm8k_dataset():
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template(
        'Template',
        model_id=MODEL_ID,
        max_length=4096,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_rewards = GSM8KAccuracyReward()(trajectories)
    brevity_rewards = GSM8KBrevityReward()(trajectories)
    total_rewards = [accuracy + brevity for accuracy, brevity in zip(accuracy_rewards, brevity_rewards)]
    return total_rewards, brevity_rewards, accuracy_rewards


@dataclass
class RolloutGroup:
    """One prompt and all of its aligned rollout and reward fields."""

    prompt: Any
    input_data: List[Dict[str, Any]]
    old_logps: List[List[float]]
    completion_lengths: List[int]
    completions: List[str]
    total_rewards: List[float]
    brevity_rewards: List[float]
    accuracy_rewards: List[float]

    @property
    def generated_tokens(self) -> int:
        return sum(self.completion_lengths)


def sample_prompt_groups(
    sampler: vLLMSampler,
    prompts: Sequence[Any],
    sampling_params: SamplingParams,
) -> List[RolloutGroup]:
    """Sample and score one complete GRPO group for every prompt."""
    expanded_prompts = [prompt for prompt in prompts for _ in range(NUM_GENERATIONS)]
    sample_responses = sampler.sample(expanded_prompts, sampling_params)

    input_data: List[Dict[str, Any]] = []
    old_logps: List[List[float]] = []
    completion_lengths: List[int] = []
    completions: List[str] = []
    for response in sample_responses:
        for sequence in response.sequences:
            if sequence.logprobs is None:
                raise RuntimeError('a sampled sequence is missing token log probabilities')
            input_data.append(sequence.new_input_feature)
            old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
            completion_lengths.append(len(sequence.tokens))
            completions.append(sequence.decoded or '')

    expected = len(prompts) * NUM_GENERATIONS
    if len(input_data) != expected:
        raise RuntimeError(f'sampler returned {len(input_data)} completions, expected {expected}')

    total_rewards, brevity_rewards, accuracy_rewards = compute_rewards(input_data)
    groups = []
    for group_index, prompt in enumerate(prompts):
        start = group_index * NUM_GENERATIONS
        end = start + NUM_GENERATIONS
        groups.append(
            RolloutGroup(
                prompt=prompt,
                input_data=input_data[start:end],
                old_logps=old_logps[start:end],
                completion_lengths=completion_lengths[start:end],
                completions=completions[start:end],
                total_rewards=total_rewards[start:end],
                brevity_rewards=brevity_rewards[start:end],
                accuracy_rewards=accuracy_rewards[start:end],
            ))
    return groups


def flatten_admitted_groups(groups: Sequence[RolloutGroup]) -> Dict[str, list]:
    """Flatten groups without changing their order or internal boundaries."""
    return {
        'input_data': [value for group in groups for value in group.input_data],
        'old_logps': [value for group in groups for value in group.old_logps],
        'completion_lengths': [value for group in groups for value in group.completion_lengths],
        'total_rewards': [value for group in groups for value in group.total_rewards],
        'brevity_rewards': [value for group in groups for value in group.brevity_rewards],
        'accuracy_rewards': [value for group in groups for value in group.accuracy_rewards],
    }


def collect_admitted_groups(
    sampler: vLLMSampler,
    prompts: Sequence[Any],
    sampling_params: SamplingParams,
    policy: GroupAdmissionPolicy,
) -> tuple[List[RolloutGroup], List[GroupAdmissionDecision], SamplingBudgetState, bool]:
    """Run initial sampling and bounded retries under one policy snapshot."""
    target_groups = len(prompts)
    if target_groups == 0:
        raise ValueError('at least one prompt is required')
    controller = SamplingBudgetController(
        SamplingBudgetConfig(
            max_extra_groups=MAX_EXTRA_GROUPS,
            max_extra_groups_per_round=target_groups,
            max_resample_rounds=MAX_RESAMPLE_ROUNDS,
            max_total_samples=(target_groups + MAX_EXTRA_GROUPS) * NUM_GENERATIONS,
            max_total_tokens=(target_groups + MAX_EXTRA_GROUPS) * NUM_GENERATIONS * MAX_NEW_TOKENS,
        ))
    state = SamplingBudgetState()
    admitted_groups: List[RolloutGroup] = []
    all_decisions: List[GroupAdmissionDecision] = []

    current_prompts = list(prompts)
    retry_queue: List[Any] = []
    round_index = 0
    exhausted = False

    while current_prompts:
        groups = sample_prompt_groups(sampler, current_prompts, sampling_params)
        decisions = [
            policy.evaluate(group.total_rewards, group.completions)
            for group in groups
        ]
        all_decisions.extend(decisions)

        for group, decision in zip(groups, decisions):
            outcome = 'admitted' if decision.admitted else decision.primary_rejection_reason
            logger.info(
                f'[Group admission] round={round_index} outcome={outcome} '
                f'reward_range={decision.reward_range:.6f} '
                f'nontrivial={decision.nontrivial_advantage_count}')
            if decision.admitted:
                admitted_groups.append(group)
            else:
                retry_queue.append(group.prompt)

        controller.observe_round(
            state,
            decisions,
            num_generations=NUM_GENERATIONS,
            generated_tokens=sum(group.generated_tokens for group in groups),
        )
        plan = controller.plan_resampling(
            state,
            target_groups=target_groups,
            num_generations=NUM_GENERATIONS,
        )
        logger.info(
            f'[Group admission] plan extra_groups={plan.extra_groups} '
            f'remaining={plan.remaining_target_groups} '
            f'effective_rate={plan.effective_rate:.4f} '
            f'exhausted={plan.exhausted} limited_by={plan.limited_by}')

        if plan.extra_groups == 0:
            exhausted = plan.exhausted
            break

        # Retry each unresolved prompt at most once per round.  When the EMA
        # requests more work than there are unresolved prompts, the next round
        # can retry them again without admitting duplicate groups for a prompt.
        retry_count = min(plan.extra_groups, len(retry_queue))
        current_prompts = retry_queue[:retry_count]
        retry_queue = retry_queue[retry_count:]
        round_index += 1

    return admitted_groups, all_decisions, state, exhausted


def admission_log_dict(
    decisions: Sequence[GroupAdmissionDecision],
    state: SamplingBudgetState,
    *,
    target_groups: int,
    admitted_groups: int,
    exhausted: bool,
) -> Dict[str, float | int]:
    initial_admitted = sum(decision.admitted for decision in decisions[:target_groups])
    initially_rejected = target_groups - initial_admitted
    recovered_groups = admitted_groups - initial_admitted
    values = {
        f'group_admission/{name}': value
        for name, value in group_admission_metrics(decisions).items()
    }
    values.update({
        'group_admission/target_group_count': target_groups,
        'group_admission/initial_effective_group_rate': initial_admitted / target_groups,
        'group_admission/final_admitted_group_count': admitted_groups,
        'group_admission/final_effective_group_rate': admitted_groups / target_groups,
        'group_admission/recovered_group_count': recovered_groups,
        'group_admission/recovery_rate': recovered_groups / initially_rejected if initially_rejected else 0.0,
        'group_admission/sampling_cost_multiplier': state.generated_groups / target_groups,
        'group_admission/budget_exhausted': int(exhausted),
    })
    return values


# ========== Main ==========
def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
    )
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
            variable_seq_lengths=True,
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model(
        ADAPTER_NAME,
        lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Template', model_id=MODEL_ID, enable_thinking=False)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': 32,
            'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Template', model_id=MODEL_ID, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    global_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=global_batch_size,
        min_batch_size=global_batch_size,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        num_samples=1,
        logprobs=1,
        temperature=1.0,
        top_p=0.95,
    )
    admission_policy = GroupAdmissionPolicy(
        GroupAdmissionConfig(
            min_reward_std=1e-6,
            min_reward_range=0.02,
            min_nontrivial_advantages=2,
            advantage_tolerance=0.01,
        ))

    optim_step = 0
    logger.info('Starting GSM8K GRPO training with budget-aware group admission')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        prompts = batch if isinstance(batch, list) else [batch]

        # Keep one rollout-policy snapshot across the initial candidates and
        # every replacement round in this batch.
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()
        admitted_groups, decisions, budget_state, exhausted = collect_admitted_groups(
            sampler,
            prompts,
            sampling_params,
            admission_policy,
        )
        admission_metrics = admission_log_dict(
            decisions,
            budget_state,
            target_groups=len(prompts),
            admitted_groups=len(admitted_groups),
            exhausted=exhausted,
        )

        if not admitted_groups:
            logger.warning(f'No complete groups admitted; skipping batch. {admission_metrics}')
            continue

        rollout_batch = flatten_admitted_groups(admitted_groups)
        metrics.accumulate(
            completion_lengths=rollout_batch['completion_lengths'],
            rewards={
                'total': rollout_batch['total_rewards'],
                'brevity': rollout_batch['brevity_rewards'],
                'accuracy': rollout_batch['accuracy_rewards'],
            },
        )
        advantages = advantage_fn(
            rollout_batch['total_rewards'],
            num_generations=NUM_GENERATIONS,
            scale='group',
        ).tolist()

        total_completions = len(rollout_batch['input_data'])
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            model.forward_backward(
                inputs=rollout_batch['input_data'][mb_start:mb_end],
                old_logps=rollout_batch['old_logps'][mb_start:mb_end],
                advantages=advantages[mb_start:mb_end],
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step % SAVE_STEPS == 0:
                model.save(f'math-grpo-group-admission-checkpoint-{optim_step}')
            if optim_step >= MAX_STEPS:
                break

        log_dict = metrics.calculate()
        log_dict.update(admission_metrics)
        log_dict.update(model.calculate_metric(is_training=True))
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('math-grpo-group-admission-final')


if __name__ == '__main__':
    main()
