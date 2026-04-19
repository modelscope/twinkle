"""GRPO training script for GSM8K dataset — MultiLoRA Megatron version.

Uses MultiLoraMegatronModel with filesystem-based LoRA sync to vLLM sampler
(no CheckpointEngineManager / GPU memory sync). Each training step saves LoRA
weights to a local directory, then passes the path to vLLMSampler via
`adapter_path` so vLLM loads the latest adapter from disk.

Model: Qwen/Qwen3.6-35B-A3B (MoE, 35B total / 3B active)
Model mesh: tp=2, ep=2, pp=2, sequence_parallel=True  (8 GPUs)
Sampler mesh: dp=2, tp=2, gpus_per_worker=2            (4 GPUs)

Uses short reasoning format: shorter thinking gets higher format reward.
Answer extracted from \\boxed{} or #### format.
"""
import os
import re
from typing import List, Tuple, Dict, Any

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import MultiLoraMegatronModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.preprocessor.llm import GSM8KProcessor

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-27B')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 5e-6))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 4))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default_0'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))
LORA_SYNC_DIR = os.environ.get('LORA_SYNC_DIR', 'output/lora_sync')

SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                 'and put your final answer within \\boxed{}.')

import swanlab
swanlab.init(
    project='twinkle',
)


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
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=4096, truncation_strategy='delete', enable_thinking=False)
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


# ========== Main ==========
def main():
    # Device groups: 8 GPUs for model (tp=2 x ep=2 x pp=2), 4 GPUs for sampler (dp=2 x tp=2)
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU',
                    gpus_per_worker=2),
    ]

    # Model mesh: tp=2, ep=2, pp=2, sequence_parallel (ref: server_config.yaml)
    model_mesh = DeviceMesh.from_sizes(
        world_size=MODEL_GPUS,
        tp_size=2,
        # ep_size=2,
        pp_size=2,
        # sequence_parallel=True,
    )
    # Sampler mesh: dp=2, tp=2
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=SAMPLER_GPUS,
        dp_size=2,
        tp_size=2,
    )

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # MoE model: use explicit target_modules to avoid fused expert weight issues with 'all-linear'
    lora_config = LoraConfig(
        target_modules='all-linear',
        # target_modules=[
        #     "self_attention.in_proj",
        #     "self_attention.out_proj",
        #     "self_attention.linear_qkv",
        #     "self_attention.linear_proj",
        #     "shared_experts.linear_fc1",
        #     "shared_experts.linear_fc2",
        #     "mlp.linear_fc1",
        #     "mlp.linear_fc2",
        # ],
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
    )

    model = MultiLoraMegatronModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
        mixed_precision='bf16',
    )

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer('default', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    model.set_loss('GRPOLoss', epsilon=0.2, adapter_name=ADAPTER_NAME)
    model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, adapter_name=ADAPTER_NAME)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': 32,
            # NOTE: To use enable_lora with qwen3.5, ensure vLLM includes PR https://github.com/vllm-project/vllm/pull/36976
            'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    # No CheckpointEngineManager — we sync via filesystem:
    #   1. model.save() writes LoRA weights to a local directory (returns checkpoint_dir)
    #   2. sampler.sample(adapter_path=...) tells vLLM to load from that directory
    # NOTE: vLLM caches LoRA by path, so each save must use a unique directory name
    #       to force vLLM to reload the updated weights.

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
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95)

    optim_step = 0
    lora_sync_path = None  # path of latest LoRA snapshot for vLLM

    logger.info('Starting GSM8K GRPO training (MultiLoraMegatron, filesystem LoRA sync)')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        metrics.reset()
        expand_prompts = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        # Save current LoRA weights to a unique directory for vLLM to load.
        # Use a step-stamped path so vLLM cache doesn't serve stale weights.
        lora_sync_path = model.save(
            f'lora-sync-step-{optim_step}',
            output_dir=LORA_SYNC_DIR,
            adapter_name=ADAPTER_NAME,
        )

        sampler.reset_prefix_cache()

        # Pass adapter_path so vLLM loads the LoRA from the local filesystem
        sample_responses = sampler.sample(
            expand_prompts,
            sampling_params,
            adapter_path=lora_sync_path,
        )

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        total_rewards, brevity_rewards, accuracy_rewards = compute_rewards(all_input_data)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'brevity': brevity_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        total_completions = len(all_input_data)
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = advantages[mb_start:mb_end]

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                micro_batch_size=MICRO_BATCH_SIZE,
                adapter_name=ADAPTER_NAME,
            )
            model.clip_grad_and_step(adapter_name=ADAPTER_NAME)
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'math-grpo-checkpoint-{optim_step}', adapter_name=ADAPTER_NAME)

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME))
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('math-grpo-final', adapter_name=ADAPTER_NAME)


if __name__ == '__main__':
    main()
