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
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '0')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 2))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 2048))
LEARNING_RATE = float(os.environ.get('LR', 1e-6))
GRPO_EPSILON = float(os.environ.get('GRPO_EPSILON', 0.2))
GRPO_BETA = float(os.environ.get('GRPO_BETA', 0.0))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 100))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 8))
TEMPERATURE = float(os.environ.get('TEMPERATURE', 1.0))
WEIGHT_SYNC_INTERVAL = int(os.environ.get('WEIGHT_SYNC_INTERVAL', 1))
ADAPTER_NAME = 'default'
DATA_NUM = int(os.environ.get('DATA_NUM', 5000))

# SwanLab is optional - only used if SWANLAB_API_KEY is set
USE_SWANLAB = True
os.environ['SWANLAB_API_KEY'] = '3hVJrk0veNB2NCm72UdJg'
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
        model_mesh = DeviceMesh.from_sizes(dp_size=MODEL_GPUS, tp_size=1, pp_size=1)
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

    # gradient_accumulation_steps=1: externally managed micro-batch loop
    model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=1,
    )
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS,
                               max_lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    else:
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

    # Use global batch size so each dataloader iteration yields enough
    # prompts for one full gradient accumulation cycle.
    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_countdown_dataset, batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh, remote_group='model', num_workers=0,
    )
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=0.95,
    )

    # ── Training loop ────────────────────────────────────────────────
    # Each dataloader iteration yields a global batch (BATCH_SIZE * GRAD_ACC prompts).
    # We sample all at once, then split into micro-batches for forward_backward.
    optim_step = 0

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        step_start = time.perf_counter()
        metrics.reset()
        timings: Dict[str, float] = {
            'weight_sync': 0.0, 'generate': 0.0, 'reward': 0.0,
            'advantage': 0.0, 'train': 0.0, 'total': 0.0,
        }

        global_prompts = batch if isinstance(batch, list) else [batch]

        # ========== 1. Weight Sync (once per optim step) ==========
        t0 = time.perf_counter()
        if optim_step % WEIGHT_SYNC_INTERVAL == 0:
            ckpt_manager.sync_weights(adapter_name=ADAPTER_NAME)
        timings['weight_sync'] = time.perf_counter() - t0

        # ========== 2. Generate (once per optim step, full global batch) ==========
        t1 = time.perf_counter()
        sample_response = sampler.sample(
            global_prompts, sampling_params, num_samples=NUM_GENERATIONS,
        )
        timings['generate'] = time.perf_counter() - t1

        # Collect all sampled data
        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sequence in sample_response.sequences:
            all_input_data.append(sequence.new_input_feature)
            all_old_logps.append(sequence.logprobs)
            all_completion_lengths.append(len(sequence.tokens))

        if not all_input_data:
            logger.warning(f"Optim step {optim_step}: No valid samples, skipping")
            continue

        # ========== 3. Compute rewards (once per optim step) ==========
        t2 = time.perf_counter()
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(all_input_data)
        timings['reward'] = time.perf_counter() - t2

        metrics.accumulate(None, None,
                           generate_time=timings['generate'],
                           weight_sync_time=timings['weight_sync'],
                           completion_lengths=all_completion_lengths,
                           rewards={
                               'total': total_rewards,
                               'format': format_rewards,
                               'accuracy': accuracy_rewards,
                           })

        # ========== 4. Compute advantages (once per optim step) ==========
        t3 = time.perf_counter()
        advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group',
        )
        advantages = advantages.tolist()
        timings['advantage'] = time.perf_counter() - t3

        frac_zero_std = 1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0

        # ========== 5. Training (split into micro-batches) ==========
        t4 = time.perf_counter()
        # Each prompt generates NUM_GENERATIONS sequences, so one micro-batch
        # is BATCH_SIZE prompts * NUM_GENERATIONS sequences.
        micro_batch_seqs = BATCH_SIZE * NUM_GENERATIONS

        for micro_idx in range(GRADIENT_ACCUMULATION_STEPS):
            start = micro_idx * micro_batch_seqs
            end = start + micro_batch_seqs
            mb_inputs = all_input_data[start:end]
            mb_old_logps = all_old_logps[start:end]
            mb_advantages = advantages[start:end]

            if not mb_inputs:
                break

            # Skip micro-batch if all advantages are zero
            if all(abs(a) < 1e-8 for a in mb_advantages):
                logger.info(f"Optim step {optim_step}, micro {micro_idx}: "
                            f"All advantages zero, skipping")
                continue

            model.forward_backward(
                inputs=mb_inputs,
                adapter_name=ADAPTER_NAME,
                advantages=mb_advantages,
                old_logps=mb_old_logps,
            )

        model.clip_grad_and_step(adapter_name=ADAPTER_NAME)
        timings['train'] = time.perf_counter() - t4

        timings['total'] = time.perf_counter() - step_start
        optim_step += 1

        # ========== 6. Log ==========
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        log_dict['train/optim_step'] = optim_step
        for k, v in timings.items():
            log_dict[f'time/{k}'] = round(v, 2)

        if USE_SWANLAB:
            swanlab.log(log_dict)
        logger.info(f"[Step {optim_step}/{MAX_STEPS}] {log_dict}")

    logger.info(f"Training completed. optim_steps={optim_step}")
    model.save('grpo-countdown-checkpoint', adapter_name=ADAPTER_NAME)

if __name__ == '__main__':
    main()
