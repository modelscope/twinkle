import gc
import os
import time
from typing import List, Tuple, Dict, Any

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, SampleResponse
from twinkle.data_format import Trajectory, InputFeature
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import VLLMSampler
from twinkle.template import Template
from twinkle.metric import CompletionRewardMetric

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen2.5-3B-Instruct')
NUM_GPUS = int(os.environ.get('NUM_GPUS', 4))
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', NUM_GPUS // 2))
SAMPLER_GPUS = NUM_GPUS - MODEL_GPUS
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 4))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 1024))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
GRPO_EPSILON = float(os.environ.get('GRPO_EPSILON', 0.2))
GRPO_BETA = float(os.environ.get('GRPO_BETA', 0.0))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 2000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
TEMPERATURE = float(os.environ.get('TEMPERATURE', 1.0))
WEIGHT_SYNC_INTERVAL = int(os.environ.get('WEIGHT_SYNC_INTERVAL', 1))
ADAPTER_NAME = 'default'
DATA_NUM = 500
USE_MEGATRON = False

# SwanLab is optional - only used if SWANLAB_API_KEY is set
USE_SWANLAB = 'SWANLAB_API_KEY' in os.environ
if USE_SWANLAB:
    import swanlab
    if USE_SWANLAB:
        swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=True)
        swanlab.init(project="ms-swift", config={
            'model_id': MODEL_ID,
            'num_gpus': NUM_GPUS,
            'model_gpus': MODEL_GPUS,
            'sampler_gpus': SAMPLER_GPUS,
            'num_generations': NUM_GENERATIONS,
            'learning_rate': LEARNING_RATE,
            'grpo_beta': GRPO_BETA,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        })
    else:
        logger.info("SWANLAB_API_KEY not set, running without experiment tracking")


def create_countdown_dataset():
    """Create Countdown Game dataset."""
    from twinkle.preprocessor import CountdownProcessor
    dataset = Dataset(DatasetMeta("ms://zouxuhong/Countdown-Tasks-3to4", data_slice=range(DATA_NUM)))
    dataset.set_template("Template", model_id=MODEL_ID, max_length=8192)
    dataset.map(CountdownProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(trajectories: List[Trajectory]) -> Tuple[List[float], List[float], List[float]]:
    """Compute format and accuracy rewards."""
    from twinkle.reward import CountDownAccuracy, FormatReward
    format_rewards = FormatReward()(trajectories, [])
    accuracy_rewards = CountDownAccuracy()(trajectories, [])
    total_rewards = [a+b for a, b in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards

def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)),
                    device_type='GPU', gpus_per_worker=1),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)),
                    device_type='GPU', gpus_per_worker=1),
    ]
    if USE_MEGATRON:
        model_mesh = DeviceMesh.from_sizes(dp_size=MODEL_GPUS, tp_size=2, pp_size=2)
    else:
        model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)
    logger.info(get_device_placement())

    lora_config = LoraConfig(
        target_modules="all-linear", r=8, lora_alpha=32, lora_dropout=0.05,
    )

    # ── Model (training) ──────────────────────────────────────────────
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
            recompute_granularity='selective',
            recompute_num_layers=None,
        )
    else:
        model = TransformersModel(
        model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
    )


    model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    model.set_optimizer('AdamW', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    model.set_lr_scheduler('LinearLR', adapter_name=ADAPTER_NAME)
    model.set_loss('GRPOLoss', adapter_name=ADAPTER_NAME,
                   epsilon=GRPO_EPSILON, beta=GRPO_BETA)
    model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME)
    model.set_template('Template', model_id=MODEL_ID, adapter_name=ADAPTER_NAME)

    sampler = VLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'load_format': 'dummy',
            'gpu_memory_utilization': 0.7,
            'max_model_len': 2048,
            'enforce_eager': True,
            'enable_sleep_mode': False,
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    dataloader = DataLoader(
        dataset=create_countdown_dataset, batch_size=BATCH_SIZE, min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh, remote_group='model', num_workers=0,
    )
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=0.95,
    )
    step = 0

    for batch in dataloader:
        if step >= MAX_STEPS:
            break

        metrics.reset()

        prompts = batch if isinstance(batch, list) else [batch]

        weight_sync_time = None
        # ========== 1. Weight Sync ==========
        if step % WEIGHT_SYNC_INTERVAL == 0:
            sync_start = time.perf_counter()
            ckpt_manager.sync_weights(adapter_name=ADAPTER_NAME)
            weight_sync_time = time.perf_counter() - sync_start

        # ========== 2. Generate ==========
        gen_start = time.perf_counter()
        sample_response = sampler.sample(prompts, sampling_params, num_samples=NUM_GENERATIONS)
        generate_time = time.perf_counter() - gen_start

        input_data : List[Dict[str, Any]] = []
        old_logps_list: List[List[float]] = []
        completion_lengths: List[int] = []

        for sequence in sample_response.sequences:
            input_data.append(sequence.new_input_feature)
            old_logps_list.append(sequence.logprobs)
            completion_lengths.append(len(sequence.tokens))

        if not input_data:
            logger.warning(f"Step {step}: No valid samples, skipping")
            step += 1
            continue

        # ========== 4. Compute rewards ==========
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(input_data)
        metrics.accumulate(None, None,
                           generate_time=generate_time,
                           weight_sync_time=weight_sync_time,
                           completion_lengths=completion_lengths,
                           rewards={
                               'total': total_rewards,
                               'format': format_rewards,
                               'accuracy': accuracy_rewards,
                           })

        # ========== 5. Compute advantages ==========
        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group')
        # Convert to list so dispatch='slice_dp' slices it in sync with inputs
        advantages = advantages.tolist()

        frac_zero_std = 1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0
        if frac_zero_std == 1.0:
            logger.info(f"Step {step}: All advantages are zero, skipping training")
            step += 1
            continue

        # ========== 6. Training step ==========
        # Pass InputFeature list directly (exact token alignment with sampler).
        # advantages and old_logps are lists, sliced in sync by dispatch.
        model.forward_backward(
            inputs=input_data,
            adapter_name=ADAPTER_NAME,
            advantages=advantages,
            old_logps=old_logps_list,
        )
        model.clip_grad_and_step(adapter_name=ADAPTER_NAME)

        from twinkle import torch_util
        gc.collect()
        torch_util.empty_cache()

        # ========== 7. Log ==========
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric())
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        if USE_SWANLAB:
            swanlab.log(log_dict)
        logger.info(log_dict)
        step += 1

    model.save('grpo-countdown-checkpoint', adapter_name=ADAPTER_NAME)


if __name__ == '__main__':
    main()
