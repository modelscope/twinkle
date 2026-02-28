import os
from typing import List, Tuple, Dict, Any

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle.metric import CompletionRewardMetric
from twinkle.preprocessor.llm import GSM8KProcessor

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS',4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 200))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16)) # global prompt-level, global completion-level batch size = BATCH_SIZE * num_generations * dp_size
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 16)) # global completion-level mini-batch-size
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2)) # per-device-micro-batch-size (completion-level), batch_size in forward_backward
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'

def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=2048)
    dataset.map(GSM8KProcessor())
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

def main():
    # set sampler and model separate to use different gpus
    device_groups = [
        DeviceGroup(name='model',ranks=list(range(MODEL_GPUS)),device_type='GPU'),
        DeviceGroup(name='sampler',ranks=list(range(MODEL_GPUS, NUM_GPUS)),device_type='GPU'),
    ]
    if USE_MEGATRON:
        model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    else:
        model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(target_modules='all-linear', r=32, lora_alpha=64, lora_dropout=0.05)

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model', mixed_precision='bf16')
    else:
        model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Template', model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 4096,
            'max_lora_rank': 32, # save as lora_config
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)

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

    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS)

    optim_step = 0
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        global_prompts = batch if isinstance(batch, list) else [batch]
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()
        sample_response = sampler.sample(
            global_prompts*NUM_GENERATIONS,
            sampling_params,
            num_samples=1,
        )

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sequence in sample_response.sequences:
            all_input_data.append(sequence.new_input_feature)
            all_old_logps.append(sequence.logprobs)
            all_completion_lengths.append(len(sequence.tokens))
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

        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        # Split completions into mini-batches and run one optim step per mini-batch.
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
            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            metrics.reset()
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('grpo-gsm8k-checkpoint')

if __name__ == '__main__':
    main()
