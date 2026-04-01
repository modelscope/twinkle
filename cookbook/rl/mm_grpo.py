"""Multimodal GRPO training demo with Qwen3.5 VL model on CLEVR dataset.

This script demonstrates on-policy GRPO (Group Relative Policy Optimization)
for visual question answering using:
  - Model: Qwen3.5-2B (vision-language model)
  - Dataset: AI-ModelScope/clevr_cogen_a_train (CLEVR visual reasoning)
  - Rewards: accuracy (answer correctness) + format (<think>/<answer> tags)
  - Template: Qwen3_5Template (handles vision token embedding merge)

Architecture:
  - Separate GPU groups for training model and vLLM sampler (Ray mode)
  - LoRA fine-tuning with NCCL weight sync between model and sampler
  - GRPO loss with PPO-style clipping (epsilon=0.2)

Usage:
    python mm_grpo.py

Environment variables:
    MODEL_ID       : Model path (default: ms://Qwen/Qwen3.5-2B)
    MODEL_GPUS     : GPUs for training model (default: 2)
    SAMPLER_GPUS   : GPUs for vLLM sampler (default: 1)
    NUM_GENERATIONS: Completions per prompt for GRPO grouping (default: 4)
    MAX_NEW_TOKENS : Max generation length (default: 4096)
    LR             : Learning rate (default: 5e-5)
    MAX_STEPS      : Total optimization steps (default: 200)
    BATCH_SIZE     : Global prompt-level batch size (default: 1)
    MINI_BATCH_SIZE: Global completion-level mini-batch size (default: 4)
    MICRO_BATCH_SIZE: Per-device micro-batch size (default: 1)
    DATA_SLICE     : Number of dataset samples to use (default: 2000)
    SAVE_STEPS     : Checkpoint save interval (default: 50)
"""
import os
from typing import Any, Dict, List, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.mm import CLEVRProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import FormatReward, MultiModalAccuracyReward
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-2B')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 2))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 4))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 5e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 200))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 4))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 1))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
DATA_SLICE = int(os.environ.get('DATA_SLICE', 2000))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 50))


def create_clevr_dataset():
    dataset = LazyDataset(
        DatasetMeta('ms://AI-ModelScope/clevr_cogen_a_train', split='train',
                    data_slice=range(DATA_SLICE)),
    )
    dataset.cast_column('image', decode=False)
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=4096)
    dataset.map(CLEVRProcessor(), remove_columns=['image', 'problem', 'solution'])
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_reward_fn = MultiModalAccuracyReward()
    format_reward_fn = FormatReward()
    accuracy_rewards = accuracy_reward_fn(trajectories)
    format_rewards = format_reward_fn(trajectories, trajectories)
    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards


def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(
            name='sampler',
            ranks=list(range(MODEL_GPUS, NUM_GPUS)),
            device_type='GPU',
            gpus_per_worker=SAMPLER_GPUS,
        ),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=1, dp_size=1)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj',
        ],
    )

    from modelscope import Qwen3_5ForConditionalGeneration
    model = TransformersModel(
        model_id=MODEL_ID,
        model_cls=Qwen3_5ForConditionalGeneration,
        device_mesh=model_mesh,
        remote_group='model',
    )

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': 8,
            'enable_lora': True,
            'limit_mm_per_prompt': {'image': 1, 'video': 0},
            'mm_processor_cache_gb': 0,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Qwen3_5Template, model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_clevr_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
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
    )

    optim_step = 0
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        global_prompts = batch if isinstance(batch, list) else [batch]

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()
        sample_responses = sampler.sample(
            global_prompts * NUM_GENERATIONS,
            sampling_params,
        )

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []
        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        total_rewards, format_rewards, accuracy_rewards = compute_rewards(all_input_data)
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
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
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'mm-grpo-clevr-checkpoint-{optim_step}')
            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            metrics.reset()
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('mm-grpo-clevr-checkpoint')


if __name__ == '__main__':
    main()
