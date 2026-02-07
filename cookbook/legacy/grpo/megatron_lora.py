"""
GRPO Training Cookbook - MegatronModel with LoRA (Standalone Mode)

Tests MegatronModel RL training with the same Countdown Game task as lora.py.

Usage:
    python cookbook/grpo/megatron_lora.py
"""

import os
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from peft import LoraConfig
import torch

import twinkle
from twinkle import DeviceMesh, DeviceGroup, Platform, get_device_placement, get_logger
from twinkle import remote_class, remote_function
from twinkle.data_format import Trajectory, Message, InputFeature
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.processor import InputProcessor
from twinkle.sampler import VLLMSampler
from twinkle.sampler.types import SamplingParams, SampleResponse
from twinkle.rl import GRPOAdvantage
import ray
from twinkle.template import Template

from transformers import AutoTokenizer
from twinkle.hub import HubOperation

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen2.5-3B-Instruct')
NUM_GPUS = 4
MODEL_GPUS = 2
SAMPLER_GPUS = 2
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 1024
LEARNING_RATE = 1e-5
GRPO_EPSILON = 0.2
GRPO_BETA = 0.0
MAX_STEPS = 20
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1
TEMPERATURE = 1.0
WEIGHT_SYNC_INTERVAL = 1
ADAPTER_NAME = 'default'


# ========== Metrics ==========
@dataclass
class TrainingMetrics:
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


# ========== Rewards ==========
def format_reward(completion: str) -> float:
    has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
    return 1.0 if (has_think and has_answer) else 0.0


def countdown_accuracy_reward(completion: str, target: int, nums: List[int]) -> float:
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
    """Process sampled responses — same logic as lora.py."""
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
                break

            seq = sequences[seq_idx]
            response_tokens = list(seq.tokens)
            response_logprobs = seq.logprobs if seq.logprobs else []
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

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
    if hasattr(result, '_is_lazy_collect') and result._is_lazy_collect:
        return result()
    if callable(result) and hasattr(result, '_get_result'):
        return result()
    return result


class SimpleWeightSync:
    """Sync weights from MegatronModel to VLLMSampler via Ray object store.

    Avoids ray.util.collective NCCL, which conflicts with Megatron's
    torch.distributed NCCL (Megatron's initialize_model_parallel creates
    NCCL communicators that are incompatible with cupy's NCCL bindings
    used by ray.util.collective).
    """

    def __init__(self, model, sampler, adapter_name: str = ''):
        self.model = model
        self.sampler = sampler
        self.adapter_name = adapter_name
        self.base_sync_done = False

    def sync_weights(self, adapter_name: str = ''):
        """Sync model weights to sampler via Ray object store."""
        adapter_name = adapter_name or self.adapter_name

        if not self.base_sync_done:
            # First sync: all base weights
            weights_dict = wait_result(
                self.model.export_weights_dict(adapter_name=adapter_name)
            )
            peft_config = None
        else:
            # Subsequent syncs: LoRA weights only
            weights_dict = wait_result(
                self.model.export_weights_dict(adapter_name=adapter_name, lora_only=True)
            )
            peft_config = wait_result(
                self.model.get_peft_config_dict(adapter_name=adapter_name)
            )

        # Load into sampler
        wait_result(self.sampler.import_weights_dict(
            weights=weights_dict,
            peft_config=peft_config,
            base_sync_done=self.base_sync_done,
        ))

        # TODO: remove this after lora sync is implemented
        # if not self.base_sync_done:
        #     self.base_sync_done = True


# ========== Main ==========
def main():
    # ── Device setup ──────────────────────────────────────────────────
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)),
                    device_type='GPU', gpus_per_worker=1),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)),
                    device_type='GPU', gpus_per_worker=1),
    ]
    # MegatronModel: DP=2, TP=1, PP=1 for 2 GPUs
    model_mesh = DeviceMesh.from_sizes(
        dp_size=MODEL_GPUS, tp_size=1, pp_size=1,
    )
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)
    logger.info(get_device_placement())

    lora_config = LoraConfig(
        target_modules="all-linear", r=8, lora_alpha=32, lora_dropout=0.05,
    )

    # ── MegatronModel (training) ──────────────────────────────────────
    model = MegatronModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
        mixed_precision='bf16',
        recompute_granularity='selective',
    )
    model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    # MegatronModel uses Megatron's distributed optimizer and scheduler
    model.set_optimizer('default', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE,
                           adapter_name=ADAPTER_NAME)
    model.set_loss('GRPOLoss', adapter_name=ADAPTER_NAME,
                   epsilon=GRPO_EPSILON, beta=GRPO_BETA)
    model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME)
    model.set_template('Template', model_id=MODEL_ID, adapter_name=ADAPTER_NAME)

    sampler = VLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'load_format': 'dummy',
            'gpu_memory_utilization': 0.3,
            'max_model_len': 2048,
            'enforce_eager': True,
            'enable_sleep_mode': False,
            'enable_lora': False, # sync lora todo
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)

    # Use SimpleWeightSync instead of CheckpointEngineManager to avoid
    # NCCL conflict between Megatron's torch.distributed and cupy NCCL.
    weight_sync = SimpleWeightSync(model, sampler, adapter_name=ADAPTER_NAME)
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
            weight_sync.sync_weights(adapter_name=ADAPTER_NAME)
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
        advantages = advantages.tolist()

        frac_zero_std = 1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0
        if frac_zero_std == 1.0:
            logger.info(f"Step {step}: All advantages are zero, skipping training")
            step += 1
            continue

        # ========== 6. Training step ==========
        # MegatronModel.forward_backward returns float loss directly
        loss = wait_result(model.forward_backward(
            inputs=input_features,
            adapter_name=ADAPTER_NAME,
            advantages=advantages,
            old_logps=old_logps_list,
        ))

        # MegatronModel: step/zero_grad/lr_step separately
        # step() stores grad_norm internally
        wait_result(model.step(adapter_name=ADAPTER_NAME))
        wait_result(model.zero_grad(adapter_name=ADAPTER_NAME))
        wait_result(model.lr_step(adapter_name=ADAPTER_NAME))

        metrics.loss = float(loss) if loss is not None else 0.0
        # grad_norm is not directly returned; it's stored in optimizer_config
        # For now, log loss only; grad_norm can be retrieved if needed
        metrics.grad_norm = 0.0

        import gc
        from twinkle.utils.framework import Torch
        gc.collect()
        Torch.empty_cache()

        # ========== 7. Log ==========
        logger.info(
            f"Step {step}: loss={metrics.loss:.6f}, grad_norm={metrics.grad_norm:.7f}, "
            f"reward={sum(metrics.rewards) / max(len(metrics.rewards), 1):.4f}, "
            f"format={sum(metrics.format_rewards) / max(len(metrics.format_rewards), 1):.2f}, "
            f"accuracy={sum(metrics.accuracy_rewards) / max(len(metrics.accuracy_rewards), 1):.2f}, "
            f"completion_len={sum(metrics.completion_lengths) / max(len(metrics.completion_lengths), 1):.1f}"
        )

        step += 1

    logger.info(f"Training completed. Total steps: {step}")


if __name__ == '__main__':
    main()
