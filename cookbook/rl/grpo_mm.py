"""GRPO training script for OlympiadBench multimodal math/physics dataset.

Supports three subsets:
- OE_MM_maths_zh_CEE: Multimodal math problems (Chinese CEE)
- OE_MM_physics_zh_CEE: Multimodal physics problems (Chinese CEE)
- OE_TO_maths_zh_CEE: Text-only math problems (Chinese CEE)
"""
import os
from typing import List, Tuple, Dict, Any

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.olympiad_bench import OlympiadBenchProcessor
from twinkle.reward.olympiad_bench import (
    OlympiadBenchAccuracyReward,
    OlympiadBenchFormatReward,
    OlympiadBenchQualityReward,
)
from twinkle.sampler import vLLMSampler

import swanlab
swanlab.init(
    project='twinkle',
)
logger = get_logger()

# Model configuration
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

# GPU configuration
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

# Training hyperparameters
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 4))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 1))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 50))

# Dataset configuration
SUBSETS = [
    'OE_MM_maths_zh_CEE',
    'OE_MM_physics_zh_CEE',
    'OE_TO_maths_zh_CEE',
]


def create_olympiad_dataset():
    """Create OlympiadBench dataset with all three subsets mixed."""
    # Create dataset with first subset
    ds = DatasetMeta(
        'ms://AI-ModelScope/OlympiadBench',
        subset_name=SUBSETS[0],
        split='train',
    )
    dataset = LazyDataset(ds)
    dataset.map(OlympiadBenchProcessor(language='zh'), dataset_meta=ds)

    # Add remaining subsets
    for subset in SUBSETS[1:]:
        ds = DatasetMeta(
            'ms://AI-ModelScope/OlympiadBench',
            subset_name=subset,
            split='train',
        )
        dataset.add_dataset(ds)
        dataset.map(OlympiadBenchProcessor(language='zh'), dataset_meta=ds)

    # Set template and preprocess
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=2048, enable_thinking=False)
    # Mix all datasets (interleave)
    dataset.mix_dataset(interleave=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Compute rewards for trajectories.

    Three core rewards, all normalized to [0, 1]:
        - Accuracy: Answer correctness (weight: 2.0)
        - Format: Answer formatting and consistency (weight: 1.0)
        - Quality: Reasoning, length, repetition (weight: 1.0)

    Returns:
        total_rewards: Weighted sum normalized to [0, 1]
        reward_dict: Individual reward components for logging
    """
    accuracy_fn = OlympiadBenchAccuracyReward()
    format_fn = OlympiadBenchFormatReward()
    quality_fn = OlympiadBenchQualityReward()

    accuracy = accuracy_fn(trajectories)
    format_r = format_fn(trajectories)
    quality = quality_fn(trajectories)

    # Weights: accuracy most important, format and quality equal
    total_rewards = [
        (2.0 * a + 1.0 * f + 1.0 * q) / 4.0
        for a, f, q in zip(accuracy, format_r, quality)
    ]

    return total_rewards, {
        'accuracy': accuracy,
        'format': format_r,
        'quality': quality,
    }


def main():
    # Device groups: model and sampler on separate GPUs
    device_groups = [
        DeviceGroup(name='model', ranks=MODEL_GPUS, device_type='GPU'),
        DeviceGroup(name='sampler', ranks=SAMPLER_GPUS, device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # LoRA configuration
    lora_config = LoraConfig(
        target_modules=['all-linear'],
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # Model setup
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )
    else:
        from transformers import Qwen3_5ForConditionalGeneration
        model = TransformersModel(
            model_id=MODEL_ID,
            model_cls=Qwen3_5ForConditionalGeneration,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)

    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=0.2, adapter_name=ADAPTER_NAME)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, adapter_name=ADAPTER_NAME, enable_thinking=False)

    # Sampler setup
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 32000,
            'max_lora_rank': 32,
            'enable_lora': True,
            'limit_mm_per_prompt': {'image': 9},  # OlympiadBench has up to 9 images
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    # Checkpoint manager
    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    # DataLoader
    GLOBAL_BATCH_SIZE = BATCH_SIZE
    dataloader = DataLoader(
        dataset=create_olympiad_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
    )

    # RL components
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1)

    optim_step = 0
    logger.info(f'Starting OlympiadBench GRPO training on subsets: {SUBSETS}')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        metrics.reset()

        # Sync weights to sampler
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        # Sample multiple completions per prompt
        sample_responses = sampler.sample(
            batch * NUM_GENERATIONS,
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

        # Compute rewards
        total_rewards, reward_dict = compute_rewards(all_input_data)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                **{k: v for k, v in reward_dict.items()},
            },
        )

        # Compute advantages
        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        # Mini-batch training
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
                model.save(f'olympiad-grpo-mixed-checkpoint-{optim_step}', adapter_name=ADAPTER_NAME)

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME))
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')
        swanlab.log(log_dict)

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('olympiad-grpo-mixed-final', adapter_name=ADAPTER_NAME)


if __name__ == '__main__':
    main()
