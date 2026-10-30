import random
import os
from typing import List, Tuple, Dict, Any

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
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
from twinkle.utils.rl_checkpoint import load_rl_checkpoint, save_rl_checkpoint

logger = get_logger()
args = CLI.from_args()

MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'
USE_MEGATRON = args.model.strategy != 'native_fsdp'

MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = args.rl.num_generations or 8
MAX_NEW_TOKENS = args.sampling.max_tokens or 4096
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
MAX_STEPS = args.training.max_steps or 200
BATCH_SIZE = args.training.batch_size or 8
MINI_BATCH_SIZE = args.training.mini_batch_size or 8
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
ADAPTER_NAME = args.lora.adapter_name or 'default'
SAVE_STEPS = args.training.save_steps or 50

def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=400)
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
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS,
                       groups=device_groups, lazy_collect=False,
                       seed=args.training.seed)

    # lora_config = LoraConfig(target_modules='all-linear', r=32, lora_alpha=64, lora_dropout=0.05)
    # Since we are training on text-only data, we avoid using 'all-linear' which would include the ViT layers.
    lora_config = LoraConfig(
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj',
        ],
        r=32, lora_alpha=64, lora_dropout=0.05,
    )
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model', mixed_precision='bf16')
    else:
        from transformers import Qwen3_5ForConditionalGeneration
        model = TransformersModel(model_id=MODEL_ID, model_cls=Qwen3_5ForConditionalGeneration, device_mesh=model_mesh, remote_group='model')

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 4496,
            'max_lora_rank': 32, # save as lora_config
            # NOTE: To use enable_lora with qwen3.5, ensure vLLM includes PR https://github.com/vllm-project/vllm/pull/36976
            # enable_lora=True used with ckpt_manager.sync_weights(merge_and_sync=False)
            # meaning only sync lora weights, if merge_and_sync=True,
            # lora will be merged into the base model and sync all weights to vLLM
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)

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


    resume_config = {
        'sampling_seed_strategy': 'driver_python_rng_n_samples_v1',
        'model_id': MODEL_ID,
        'use_megatron': USE_MEGATRON,
        'model_gpus': MODEL_GPUS,
        'sampler_gpus': SAMPLER_GPUS,
        'global_batch_size': GLOBAL_BATCH_SIZE,
        'mini_batch_size': MINI_BATCH_SIZE,
        'micro_batch_size': MICRO_BATCH_SIZE,
        'num_generations': NUM_GENERATIONS,
        'max_new_tokens': MAX_NEW_TOKENS,
        'max_steps': MAX_STEPS,
        'learning_rate': LEARNING_RATE,
        'seed': args.training.seed,
        'adapter_name': ADAPTER_NAME,
    }

    optim_step, pending = 0, None
    if args.training.resume_from_checkpoint:
        if args.training.resume_only_model or args.training.ignore_data_skip:
            raise ValueError(
                'GRPO continuation requires full training and data state'
            )
        optim_step, pending = load_rl_checkpoint(
            model, dataloader,
            args.training.resume_from_checkpoint,
            adapter_name=ADAPTER_NAME,
            config=resume_config,
        )
        logger.info(f'Resumed GRPO at optimizer step {optim_step}')

    if optim_step >= MAX_STEPS:
        logger.info(f'Checkpoint already reached max_steps={MAX_STEPS}')
        return

    def save_checkpoint(name, remaining):
        path = save_rl_checkpoint(
            model, dataloader,
            name=name,
            output_dir=args.training.output_dir,
            adapter_name=ADAPTER_NAME,
            optim_step=optim_step,
            pending=remaining,
            config=resume_config,
        )
        logger.info(f'Saved complete RL checkpoint: {path}')

    logger.info(get_device_placement())
    batches = iter(dataloader)

    while optim_step < MAX_STEPS:
        if pending is None:
            try:
                batch = next(batches)
            except StopIteration:
                break
            metrics.reset()
            global_prompts = batch if isinstance(batch, list) else [batch]
            # enable_lora=True used with ckpt_manager.sync_weights(merge_and_sync=False)
            # meaning only sync lora weights, if merge_and_sync=True,
            # lora will be merged into the base model and sync all weights to vLLM
            ckpt_manager.sync_weights(merge_and_sync=False)
            sampler.reset_prefix_cache()
            # The driver Python RNG is saved in the RL checkpoint.
            rollout_seed = random.getrandbits(63)
            rollout_params = SamplingParams(
                max_tokens=MAX_NEW_TOKENS,
                num_samples=NUM_GENERATIONS,
                logprobs=1,
                seed=rollout_seed,
            )
            sample_responses = sampler.sample(
                global_prompts,
                rollout_params,
            )

            all_input_data: List[Dict[str, Any]] = []
            all_old_logps: List[List[float]] = []
            all_completion_lengths: List[int] = []

            for sample_response in sample_responses:
                for sequence in sample_response.sequences:
                    all_input_data.append(sequence.new_input_feature)
                    all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
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

            pending = {
                'inputs': all_input_data,
                'old_logps': all_old_logps,
                'advantages': advantages,
                'metrics': metrics.calculate(),
            }

        all_input_data = pending['inputs']
        all_old_logps = pending['old_logps']
        advantages = pending['advantages']

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

            remaining = None if mb_end == total_completions else {
                'inputs': all_input_data[mb_end:],
                'old_logps': all_old_logps[mb_end:],
                'advantages': advantages[mb_end:],
                'metrics': pending['metrics'],
            }

            log_dict = dict(pending['metrics'])
            log_dict.update(model.calculate_metric(
                is_training=True, adapter_name=ADAPTER_NAME,
            ))
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

            if optim_step % SAVE_STEPS == 0:
                save_checkpoint(
                    f'grpo-gsm8k-checkpoint-{optim_step}', remaining,
                )

            if optim_step >= MAX_STEPS:
                pending = remaining
                break
        else:
            pending = None

    logger.info(f'Training completed. optim_steps={optim_step}')
    save_checkpoint(
        f'grpo-gsm8k-checkpoint-{optim_step}-final', pending,
    )


if __name__ == '__main__':
    main()
