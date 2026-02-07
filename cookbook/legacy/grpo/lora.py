"""
GRPO Training Cookbook - Standalone Mode with LoRA

This cookbook demonstrates GRPO training using TransformersModel and VLLMSampler
in standalone mode (model and sampler on different GPUs with NCCL weight sync).

Task: Countdown Game
- Given numbers [a, b, c, d], find an equation using +, -, *, / that equals target
- Rewards: format reward (<think>/<answer> tags) + accuracy reward (correct equation)

Usage:
    SWANLAB_API_KEY=xxx python cookbook/grpo/lora.py
"""

import os
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from peft import LoraConfig
import torch

import twinkle
from twinkle import DeviceMesh, DeviceGroup, Platform, get_device_placement, get_logger
from twinkle import remote_class, remote_function
from twinkle.data_format import Trajectory, Message, InputFeature
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import VLLMSampler
from twinkle.sampler.types import SamplingParams, SampleResponse
from twinkle.rl import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.template import Template

from transformers import AutoTokenizer
from twinkle.hub import HubOperation

# SwanLab is optional - only used if SWANLAB_API_KEY is set
USE_SWANLAB = 'SWANLAB_API_KEY' in os.environ
if USE_SWANLAB:
    import swanlab

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


# ========== Metrics ==========
@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    generate_time: float = 0.0
    weight_sync_time: float = 0.0
    rewards: List[float] = field(default_factory=list)
    format_rewards: List[float] = field(default_factory=list)
    accuracy_rewards: List[float] = field(default_factory=list)
    completion_lengths: List[int] = field(default_factory=list)
    loss: float = 0.0
    grad_norm: float = 0.0

    def reset(self):
        self.generate_time = 0.0
        self.weight_sync_time = 0.0
        self.rewards = []
        self.format_rewards = []
        self.accuracy_rewards = []
        self.completion_lengths = []
        self.loss = 0.0
        self.grad_norm = 0.0

    def to_log_dict(self, step: int) -> Dict[str, float]:
        log_dict = {
            'step': step,
            'profiling/Time taken: GRPOTrainer._move_model_to_vllm': self.weight_sync_time,
            'profiling/Time taken: GRPOTrainer.generate': self.generate_time,
            'train/loss': self.loss,
            'train/grad_norm': self.grad_norm,
        }
        if self.rewards:
            log_dict['train/reward'] = sum(self.rewards) / len(self.rewards)
            log_dict['train/reward_std'] = torch.tensor(self.rewards).std().item() if len(self.rewards) > 1 else 0.0
        if self.format_rewards:
            log_dict['train/rewards/Format/mean'] = sum(self.format_rewards) / len(self.format_rewards)
        if self.accuracy_rewards:
            log_dict['train/rewards/CountdownORM/mean'] = sum(self.accuracy_rewards) / len(self.accuracy_rewards)
        if self.completion_lengths:
            log_dict['train/completions/mean_length'] = sum(self.completion_lengths) / len(self.completion_lengths)
        return log_dict


# ========== Rewards ==========
def format_reward(completion: str) -> float:
    """Format reward: checks <think> and <answer> tags."""
    has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
    return 1.0 if (has_think and has_answer) else 0.0


def countdown_accuracy_reward(completion: str, target: int, nums: List[int]) -> float:
    """Accuracy reward: checks if equation is correct."""
    try:
        match = re.search(r'<answer>(.*?)<\/answer>', completion)
        if match is None:
            return 0.0
        equation = match.group(1).strip()
        if '=' in equation:
            equation = equation.split('=')[0]
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        if not re.match(r'^[\d+\-*/().\s]+$', equation):
            return 0.0
        result = eval(equation, {'__builtins__': None}, {})
        return 1.0 if abs(float(result) - float(target)) < 1e-5 else 0.0
    except Exception:
        return 0.0


# ========== Dataset ==========
def create_countdown_dataset():
    """Create Countdown Game dataset."""
    from twinkle.preprocessor import CountdownProcessor
    dataset = Dataset(DatasetMeta("ms://zouxuhong/Countdown-Tasks-3to4", data_slice=range(50000)))
    dataset.set_template("Template", model_id=MODEL_ID, max_length=8192)
    dataset.map(CountdownProcessor())
    return dataset


# ========== Sample Processing ==========
def process_samples(
    prompts: List[Trajectory],
    sample_response: SampleResponse,
    tokenizer,
    num_generations: int,
    template: Template,
) -> Tuple[List[Trajectory], List[InputFeature], List[List[float]], List[int]]:
    """Process sampled responses.

    Builds ``InputFeature`` directly by concatenating prompt token ids with
    the sampler's raw response token ids, avoiding decode/re-encode drift.

    Returns:
        (trajectories, input_features, old_logps_list, completion_lengths)
    """
    trajectories: List[Trajectory] = []
    input_features: List[InputFeature] = []
    old_logps_list: List[List[float]] = []
    completion_lengths: List[int] = []

    sequences = sample_response.sequences
    prompt_ids_cache: Dict[int, List[int]] = {}

    for i, prompt in enumerate(prompts):
        if i not in prompt_ids_cache:
            prompt_messages = [
                dict(msg) for msg in prompt.get('messages', [])
                if not (msg.get('role') == 'assistant'
                        and not msg.get('content', '').strip())
            ]
            encoded = tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True,
            )
            if hasattr(encoded, 'tolist'):
                encoded = encoded.tolist()
            prompt_ids_cache[i] = list(encoded)

        prompt_ids = prompt_ids_cache[i]

        for j in range(num_generations):
            seq_idx = i * num_generations + j
            if seq_idx >= len(sequences):
                logger.warning(
                    f"Expected {len(prompts) * num_generations} sequences, "
                    f"got {len(sequences)}"
                )
                break

            seq = sequences[seq_idx]
            response_tokens = list(seq.tokens)
            response_logprobs = seq.logprobs if seq.logprobs else []
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Trajectory (for reward computation only)
            messages = [
                msg for msg in prompt.get('messages', [])
                if not (msg.get('role') == 'assistant'
                        and not msg.get('content', '').strip())
            ]
            messages.append(Message(role='assistant', content=response_text))
            trajectories.append(Trajectory(
                messages=messages,
                user_data=prompt.get('user_data', []),
            ))

            # InputFeature (exact token alignment with sampler)
            input_ids = prompt_ids + response_tokens
            labels = [-100] * len(prompt_ids) + response_tokens
            input_feature = InputFeature(
                input_ids=np.array(input_ids),
                labels=np.array(labels),
            )
            input_feature = template._invoke_post_pipeline([input_feature])
            input_features.append(input_feature[0])

            old_logps_list.append(response_logprobs)
            completion_lengths.append(len(response_tokens))

    return trajectories, input_features, old_logps_list, completion_lengths


def compute_rewards(trajectories: List[Trajectory]) -> Tuple[List[float], List[float], List[float]]:
    """Compute format and accuracy rewards."""
    total_rewards, format_rewards, accuracy_rewards = [], [], []
    for traj in trajectories:
        messages = traj.get('messages', [])
        completion = ""
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                completion = msg.get('content', '')
                break
        user_data = traj.get('user_data', [{}])
        data = user_data[0] if isinstance(user_data, list) and user_data else {}
        target = data.get('target', 0)
        nums = data.get('nums', [])
        fmt_reward = format_reward(completion)
        acc_reward = countdown_accuracy_reward(completion, target, nums)
        format_rewards.append(fmt_reward)
        accuracy_rewards.append(acc_reward)
        total_rewards.append(fmt_reward + acc_reward)
    return total_rewards, format_rewards, accuracy_rewards


def wait_result(result):
    """Wait for lazy collect result if needed."""
    if hasattr(result, '_is_lazy_collect') and result._is_lazy_collect:
        return result()
    if callable(result) and hasattr(result, '_get_result'):
        return result()
    return result


# ========== Main ==========
def main():
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

    # ── Device setup ──────────────────────────────────────────────────
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)),
                    device_type='GPU', gpus_per_worker=1),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)),
                    device_type='GPU', gpus_per_worker=1),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)
    logger.info(get_device_placement())

    lora_config = LoraConfig(
        target_modules="all-linear", r=8, lora_alpha=32, lora_dropout=0.05,
    )

    # ── Model (training) ──────────────────────────────────────────────
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
    dataset = create_countdown_dataset()
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE,
        device_mesh=model_mesh, remote_group='model', num_workers=0,
    )
    model_path = HubOperation.download_model(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    advantage_fn = GRPOAdvantage()
    metrics = TrainingMetrics()

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=0.95,
    )
    step = 0

    for batch in dataloader:
        if step >= MAX_STEPS:
            break

        metrics.reset()

        if callable(batch):
            batch = batch()
        prompts = batch if isinstance(batch, list) else [batch]

        # ========== 1. Weight Sync ==========
        if step % WEIGHT_SYNC_INTERVAL == 0:
            sync_start = time.perf_counter()
            ckpt_manager.sync_weights(adapter_name=ADAPTER_NAME)
            metrics.weight_sync_time = time.perf_counter() - sync_start

        # ========== 2. Generate ==========
        gen_start = time.perf_counter()
        sample_response = wait_result(
            sampler.sample(prompts, sampling_params, num_samples=NUM_GENERATIONS)
        )
        metrics.generate_time = time.perf_counter() - gen_start

        # ========== 3. Process samples ==========
        template = sampler._get_template(adapter_name=ADAPTER_NAME)
        trajectories, input_features, old_logps_list, completion_lengths = \
            process_samples(prompts, sample_response, tokenizer, NUM_GENERATIONS, template)

        if not trajectories:
            logger.warning(f"Step {step}: No valid samples, skipping")
            step += 1
            continue

        metrics.completion_lengths = completion_lengths

        # ========== 4. Compute rewards ==========
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(trajectories)
        metrics.rewards = total_rewards
        metrics.format_rewards = format_rewards
        metrics.accuracy_rewards = accuracy_rewards

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
        loss = wait_result(model.forward_backward(
            inputs=input_features,
            adapter_name=ADAPTER_NAME,
            advantages=advantages,
            old_logps=old_logps_list,
        ))

        grad_norm = wait_result(model.clip_grad_and_step(adapter_name=ADAPTER_NAME))
        metrics.loss = float(loss) if loss else 0.0
        if isinstance(grad_norm, list):
            grad_norm = grad_norm[0]
        metrics.grad_norm = float(grad_norm) if isinstance(grad_norm, (int, float)) else 0.0

        from twinkle.utils.framework import Torch
        import gc
        gc.collect()
        Torch.empty_cache()

        # ========== 7. Log ==========
        log_dict = metrics.to_log_dict(step)
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        if USE_SWANLAB:
            swanlab.log(log_dict)

        logger.info(
            f"Step {step}: loss={metrics.loss:.6f}, grad_norm={metrics.grad_norm:.7f}, "
            f"reward={log_dict.get('train/reward', 0):.4f}, "
            f"format={log_dict.get('train/rewards/Format/mean', 0):.2f}, "
            f"accuracy={log_dict.get('train/rewards/CountdownORM/mean', 0):.2f}, "
            f"completion_len={log_dict.get('train/completions/mean_length', 0):.1f}"
        )

        step += 1

    logger.info(f"Training completed. Total steps: {step}")
    wait_result(model.save('grpo-countdown-checkpoint', adapter_name=ADAPTER_NAME))
    if USE_SWANLAB:
        swanlab.finish()


if __name__ == '__main__':
    main()
