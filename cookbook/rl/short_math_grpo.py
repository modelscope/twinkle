"""GRPO training script for GSM8K dataset.

Converted from the Tinker client version to Ray-based training.
Uses short reasoning format: shorter thinking gets higher format reward.
Answer extracted from \\boxed{} or #### format.
"""
import math
import os
import re
from typing import List, Tuple, Dict, Any, Optional

import swanlab
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle.preprocessor.llm import GSM8KProcessor

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))

GSM8K_MAX_LENGTH = int(os.environ.get('GSM8K_MAX_LENGTH', 4096))

KL_BETA = float(os.environ.get('KL_BETA', 0.0))
ENTROPY_COEF = float(os.environ.get('ENTROPY_COEF', 0.0))
CISPO_EPS_LOW = float(os.environ.get('CISPO_EPS_LOW', 0.2))
CISPO_EPS_HIGH = float(os.environ.get('CISPO_EPS_HIGH', 0.2))
HIGH_KL_TOPK = int(os.environ.get('HIGH_KL_TOPK', 0))

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
                if length <= 300:
                    rewards.append(1.0)
                else:
                    rewards.append(max(0.0, 1.0 - (length - 300) / 3000))
        return rewards


# ========== Dataset ==========
def create_gsm8k_dataset():
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=GSM8K_MAX_LENGTH,
                         truncation_strategy='delete', enable_thinking=False)
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_reward_fn = GSM8KAccuracyReward()
    brevity_reward_fn = GSM8KBrevityReward()

    accuracy_rewards = accuracy_reward_fn(trajectories)
    brevity_rewards = brevity_reward_fn(trajectories)
    total_rewards = [a + b for a, b in zip(accuracy_rewards, brevity_rewards)]
    return total_rewards, brevity_rewards, accuracy_rewards


# ========== Diagnostics ==========
_LEADING_NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def _coerce_for_swanlab(log_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Cast string-valued metrics to float for swanlab line charts."""
    coerced: Dict[str, Any] = {}
    for k, v in log_dict.items():
        if isinstance(v, bool) or isinstance(v, (int, float)):
            coerced[k] = v
            continue
        if isinstance(v, str):
            m = _LEADING_NUMBER_RE.search(v)
            if m:
                try:
                    coerced[k] = float(m.group())
                    continue
                except ValueError:
                    pass
        coerced[k] = v
    return coerced


def _logp_split_diagnostics(
    accuracy_rewards: List[float],
    old_logps: List[List[float]],
) -> Dict[str, float]:
    """Split mean old-logp by accuracy outcome (pos vs zero)."""
    out: Dict[str, float] = {}
    if not accuracy_rewards or not old_logps:
        return out
    per_traj_mean = [(sum(lp) / len(lp)) if lp else 0.0 for lp in old_logps]
    pos_logp = [m for m, a in zip(per_traj_mean, accuracy_rewards) if a > 0]
    zero_logp = [m for m, a in zip(per_traj_mean, accuracy_rewards) if a <= 0]
    out['acc_correct_rate'] = len(pos_logp) / len(accuracy_rewards)
    out['mean_old_logp_acc_pos'] = (sum(pos_logp) / len(pos_logp)) if pos_logp else 0.0
    out['mean_old_logp_acc_zero'] = (sum(zero_logp) / len(zero_logp)) if zero_logp else 0.0
    out['policy_confidence_acc_pos'] = math.exp(out['mean_old_logp_acc_pos'])
    out['policy_confidence_acc_zero'] = math.exp(out['mean_old_logp_acc_zero'])
    return out


# ========== Main ==========
def main():
    swanlab.init(project='twinkle')

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

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=CISPO_EPS_LOW, epsilon_high=CISPO_EPS_HIGH,
                   beta=KL_BETA, entropy_coef=ENTROPY_COEF)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    model.add_metric('GRPOMetric', is_training=True,
                     epsilon=CISPO_EPS_LOW, epsilon_high=CISPO_EPS_HIGH,
                     top_k_kl=HIGH_KL_TOPK)

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
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)
    rollout_template = Qwen3_5Template(MODEL_ID, max_length=GSM8K_MAX_LENGTH, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95,
        include_stop_str_in_output=True)

    optim_step = 0
    logger.info('Starting GSM8K GRPO training (short reasoning)')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        batch_step = optim_step

        metrics.reset()
        expand_prompts = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        sample_responses = sampler.sample(expand_prompts, sampling_params)

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        total_rewards, brevity_rewards, accuracy_rewards = compute_rewards(all_input_data)

        rollout_advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        all_acc_labels: List[bool] = [a > 0 for a in accuracy_rewards]
        n_pos = sum(1 for p in all_acc_labels if p)
        n_neg = sum(1 for p in all_acc_labels if not p)
        pos_with_neg_adv = sum(1 for p, a in zip(all_acc_labels, rollout_advantages) if p and a < 0)
        neg_with_pos_adv = sum(1 for p, a in zip(all_acc_labels, rollout_advantages) if not p and a > 0)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'brevity': brevity_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        total_completions = len(all_input_data)
        aligned_completions = (total_completions // MODEL_GPUS) * MODEL_GPUS
        if aligned_completions < total_completions:
            logger.info(
                '[dp-align] dropping %d tail sample(s): total=%d -> aligned=%d (dp=%d)',
                total_completions - aligned_completions,
                total_completions, aligned_completions, MODEL_GPUS)

        for mb_start in range(0, aligned_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, aligned_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = rollout_advantages[mb_start:mb_end]
            mb_pos_mask = all_acc_labels[mb_start:mb_end]

            ref_logps = None
            if KL_BETA > 0.0:
                ref_outputs = model.forward_only(inputs=mb_inputs, disable_lora=True)
                ref_logps = ref_outputs.get('logps') if isinstance(ref_outputs, dict) else getattr(ref_outputs, 'logps', None)

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                ref_logps=ref_logps,
                positive_mask=mb_pos_mask,
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'math-grpo-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict.update(_logp_split_diagnostics(accuracy_rewards, all_old_logps))
        log_dict['pos_neg_adv_rate'] = pos_with_neg_adv / n_pos if n_pos else 0.0
        log_dict['neg_pos_adv_rate'] = neg_with_pos_adv / n_neg if n_neg else 0.0
        log_dict['adv_max'] = max(rollout_advantages) if rollout_advantages else 0.0
        log_dict['adv_min'] = min(rollout_advantages) if rollout_advantages else 0.0

        _hk = log_dict.pop('_high_kl_records', None)
        if _hk:
            _tok = rollout_template.tokenizer
            for r in _hk:
                gsi = r.get('gsi')
                try:
                    tok_text = _tok.decode([r['token_id']])
                except Exception:
                    tok_text = None
                logger.info(
                    '[high-kl] step=%d gsi=%s pos=%s tok=%r kl=%.4f r=%.4f lp_new=%.4f lp_old=%.4f',
                    batch_step, gsi, r.get('pos'), tok_text,
                    r.get('kl'), r.get('ratio'), r.get('logp_new'), r.get('logp_old'))

        swanlab.log(_coerce_for_swanlab(log_dict), step=batch_step)
        metrics.reset()
        logger.info(f'[Step {batch_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('math-grpo-final')


if __name__ == '__main__':
    main()
