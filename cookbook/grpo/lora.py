"""
GRPO Training Cookbook - Hybrid Mode with LoRA

This cookbook demonstrates GRPO training using TransformersModel and VLLMSampler
in hybrid mode (model and sampler colocated on same GPUs with IPC weight sync).

Task: Countdown Game
- Given numbers [a, b, c, d], find an equation using +, -, *, / that equals target
- Rewards: format reward (<think>/<answer> tags) + accuracy reward (correct equation)

Reference: swift/docs/source/BestPractices/GRPO.md

Usage:
    SWANLAB_API_KEY=xxx python lora.py
"""

import os
import re
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from peft import LoraConfig
import torch

import twinkle
from twinkle import DeviceMesh, DeviceGroup, Platform, get_device_placement, get_logger
from twinkle.data_format import Trajectory, Message
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import VLLMSampler
from twinkle.sampler.types import SamplingParams, SampleResponse
from twinkle.rl import GRPOAdvantage
from twinkle.weight_loader import IPCWeightLoader
from twinkle.template import Template

from transformers import AutoTokenizer
from twinkle.hub import HubOperation

import swanlab

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen2.5-3B-Instruct')
NUM_GPUS = int(os.environ.get('NUM_GPUS', 4))
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 1024))
LEARNING_RATE = float(os.environ.get('LR', 5e-7))
GRPO_EPSILON = float(os.environ.get('GRPO_EPSILON', 0.2))
GRPO_BETA = float(os.environ.get('GRPO_BETA', 0.001))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 2000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
TEMPERATURE = float(os.environ.get('TEMPERATURE', 1.0))
WEIGHT_SYNC_INTERVAL = int(os.environ.get('WEIGHT_SYNC_INTERVAL', 1))

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first thinks about the reasoning process "
    "in the mind and then provides the user with the answer."
)
ADAPTER_NAME = 'default'


# ========== Profiling Context ==========
@contextmanager
def profiling_context(name: str):
    """Context manager for timing and logging."""
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time
    swanlab.log({f'profiling/Time taken: GRPOTrainer.{name}': duration})


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
        
        if self.format_rewards:
            log_dict['train/rewards/FormatORM/mean'] = sum(self.format_rewards) / len(self.format_rewards)
            log_dict['train/rewards/FormatORM/std'] = torch.tensor(self.format_rewards).std().item() if len(self.format_rewards) > 1 else 0.0
        
        if self.accuracy_rewards:
            log_dict['train/rewards/CountdownORM/mean'] = sum(self.accuracy_rewards) / len(self.accuracy_rewards)
            log_dict['train/rewards/CountdownORM/std'] = torch.tensor(self.accuracy_rewards).std().item() if len(self.accuracy_rewards) > 1 else 0.0
        
        if self.completion_lengths:
            log_dict['train/completions/mean_length'] = sum(self.completion_lengths) / len(self.completion_lengths)
        
        return log_dict


# ========== Reward Functions ==========
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
        
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
            return 0.0
        
        result = eval(equation, {'__builtins__': None}, {})
        return 1.0 if abs(float(result) - float(target)) < 1e-5 else 0.0
    except Exception:
        return 0.0


def compute_rewards(trajectories: List[Trajectory]) -> Tuple[List[float], List[float], List[float]]:
    """Compute format and accuracy rewards from trajectories.
    
    Args:
        trajectories: List of trajectories with 'messages' and 'user_data'.
        
    Returns:
        Tuple of (total_rewards, format_rewards, accuracy_rewards).
    """
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


# ========== Dataset ==========
def create_countdown_dataset():
    """Create Countdown Game dataset."""
    def countdown_processor(row: Dict[str, Any]) -> Dict[str, Any]:
        nums = row.get('nums', [])
        target = row.get('response', row.get('target', 0))
        
        query = f"""Using the numbers {nums}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags,
for example <answer> (1 + 2) / 3 * 4 = 4 </answer>."""
        
        return {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': ''},
            ],
            'user_data': [{'target': target, 'nums': nums}],
        }
    
    dataset = Dataset(DatasetMeta("ms://zouxuhong/Countdown-Tasks-3to4", data_slice=range(5000)))
    dataset.set_template("Template", model_id=MODEL_ID, max_length=2048)
    dataset.map(countdown_processor)
    return dataset


# ========== Sample Processing ==========
def process_samples(
    prompts: List[Trajectory],
    sample_response: SampleResponse,
    tokenizer,
    num_generations: int,
) -> List[Tuple[Trajectory, List[float], int]]:
    """
    Process sampled responses into (trajectory, old_logps, length) tuples.
    
    Args:
        prompts: List of original prompts (P prompts).
        sample_response: Response containing sequences (P * num_generations sequences).
        tokenizer: Tokenizer for decoding.
        num_generations: Number of generations per prompt (G).
        
    Returns:
        List of (trajectory, old_logps, length) tuples.
        The list has P * G entries, organized as:
        [prompt0_gen0, prompt0_gen1, ..., prompt1_gen0, prompt1_gen1, ...]
    """
    results = []
    sequences = sample_response.sequences
    
    # Sequences are organized as: for each prompt, num_generations sequences
    for i, prompt in enumerate(prompts):
        for j in range(num_generations):
            seq_idx = i * num_generations + j
            if seq_idx >= len(sequences):
                logger.warning(f"Expected {len(prompts) * num_generations} sequences, got {len(sequences)}")
                break
            
            seq = sequences[seq_idx]
            response_tokens = seq.tokens
            response_logprobs = seq.logprobs if seq.logprobs else []
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Build trajectory with response
            messages = []
            for msg in prompt.get('messages', []):
                # Skip empty assistant placeholder
                if msg.get('role') == 'assistant' and not msg.get('content', '').strip():
                    continue
                messages.append(msg)
            messages.append(Message(role='assistant', content=response_text))
            
            # Copy user_data from prompt for reward computation
            traj = Trajectory(messages=messages, user_data=prompt.get('user_data', []))
            results.append((traj, response_logprobs, len(response_tokens)))
    
    return results


def wait_result(result):
    """Wait for lazy collect result if needed."""
    if hasattr(result, '_is_lazy_collect') and result._is_lazy_collect:
        return result()
    if callable(result) and hasattr(result, '_get_result'):
        return result()
    return result


# ========== Main ==========
def main():
    # SwanLab setup
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=True)
    swanlab.init(project="ms-swift", config={
        'model_id': MODEL_ID,
        'num_gpus': NUM_GPUS,
        'num_generations': NUM_GENERATIONS,
        'learning_rate': LEARNING_RATE,
        'grpo_beta': GRPO_BETA,
        'batch_size': BATCH_SIZE,
    })
    
    # Hybrid mode: model and sampler on same GPUs
    device_groups = [
        DeviceGroup(name='hybrid', ranks=list(range(NUM_GPUS)), device_type='GPU', gpus_per_worker=1),
    ]
    device_mesh = DeviceMesh.from_sizes(world_size=NUM_GPUS, dp_size=NUM_GPUS)
    
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)
    logger.info(get_device_placement())
    
    # Dataset
    dataset = create_countdown_dataset()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        device_mesh=device_mesh,
        remote_group='hybrid',
        num_workers=0,
    )
    
    # Actor model
    actor_model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=device_mesh,
        remote_group='hybrid',
    )
    lora_config = LoraConfig(target_modules="all-linear")
    actor_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=8)
    actor_model.set_optimizer('AdamW', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    actor_model.set_lr_scheduler('LinearLR', adapter_name=ADAPTER_NAME)
    actor_model.set_loss('GRPOLoss', adapter_name=ADAPTER_NAME, epsilon=GRPO_EPSILON, beta=GRPO_BETA)
    actor_model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME)
    actor_model.set_template('Template', model_id=MODEL_ID, adapter_name=ADAPTER_NAME)
    logger.info(actor_model.get_train_configs(adapter_name=ADAPTER_NAME))
    
    # Sampler (with sleep mode for hybrid)
    sampler = VLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.4,
            'max_model_len': 8192,
            'enable_sleep_mode': True,
            'load_format': 'dummy',  # Weights will be synced via IPC
        },
        device_mesh=device_mesh,
        remote_group='hybrid',
    )
    sampler.set_template('Template', model_id=MODEL_ID)
    
    # Tokenizer
    model_path = HubOperation.download_model(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Weight loader for IPC sync
    weight_loader = IPCWeightLoader(actor_model, sampler)
    
    # Advantage calculator
    advantage_fn = GRPOAdvantage()
    
    # Metrics
    metrics = TrainingMetrics()
    
    # Sampling params (num_samples passed to sample() method)
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )
    
    logger.info(f"Starting training for {MAX_STEPS} steps")
    step = 0
    
    for batch in dataloader:
        if step >= MAX_STEPS:
            break
        
        metrics.reset()
        
        if callable(batch):
            batch = batch()
        prompts = batch if isinstance(batch, list) else [batch]
        
        # ========== 1. Weight Sync (before generation) ==========
        if step % WEIGHT_SYNC_INTERVAL == 0:
            with profiling_context('_move_model_to_vllm'):
                weight_loader.load_weights(adapter_name=ADAPTER_NAME)
        
        # ========== 2. Generate samples ==========
        # Pass num_samples to sample() to generate NUM_GENERATIONS completions per prompt
        with profiling_context('generate'):
            sampler.wake_up()
            
            gen_start = time.perf_counter()
            sample_response = wait_result(
                sampler.sample(prompts, sampling_params, num_samples=NUM_GENERATIONS)
            )
            metrics.generate_time = time.perf_counter() - gen_start
            
            sampler.sleep()
        
        # ========== 3. Process samples ==========
        samples = process_samples(prompts, sample_response, tokenizer, NUM_GENERATIONS)
        
        if not samples:
            logger.warning(f"Step {step}: No valid samples, skipping")
            step += 1
            continue
        
        trajectories = [s[0] for s in samples]
        old_logps_list = [s[1] for s in samples]
        completion_lengths = [s[2] for s in samples]
        
        metrics.completion_lengths = completion_lengths
        
        # ========== 4. Compute rewards ==========
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(trajectories)
        
        metrics.rewards = total_rewards
        metrics.format_rewards = format_rewards
        metrics.accuracy_rewards = accuracy_rewards
        
        # ========== 5. Compute advantages and add to trajectories ==========
        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group')
        
        # Add advantages to trajectories (GRPOLoss extracts from trajectory['advantages'])
        for i, traj in enumerate(trajectories):
            traj['advantages'] = float(advantages[i])
        
        # Skip if all advantages are zero
        if all(abs(adv) < 1e-8 for adv in advantages):
            logger.info(f"Step {step}: All advantages are zero, skipping training")
            step += 1
            continue
        
        # ========== 6. Training step ==========
        # GRPOLoss expects:
        # - inputs: trajectories (with 'advantages' field)
        # - old_logps: log probs from sampling policy
        # - trajectories: passed as kwarg for advantage extraction
        loss = wait_result(actor_model.forward_backward(
            inputs=trajectories,
            adapter_name=ADAPTER_NAME,
            trajectories=trajectories,  # GRPOLoss extracts advantages from here
            old_logps=old_logps_list,   # Log probs from sampling
        ))
        
        grad_norm = wait_result(actor_model.clip_grad_and_step(adapter_name=ADAPTER_NAME))
        
        metrics.loss = float(loss) if loss else 0.0
        metrics.grad_norm = float(grad_norm) if grad_norm else 0.0
        
        # ========== 7. Log metrics ==========
        log_dict = metrics.to_log_dict(step)
        swanlab.log(log_dict)
        
        logger.info(
            f"Step {step}: loss={metrics.loss:.4f}, grad_norm={metrics.grad_norm:.4f}, "
            f"reward={log_dict.get('train/reward', 0):.4f}, "
            f"format={log_dict.get('train/rewards/FormatORM/mean', 0):.2f}, "
            f"accuracy={log_dict.get('train/rewards/CountdownORM/mean', 0):.2f}, "
            f"completion_len={log_dict.get('train/completions/mean_length', 0):.1f}"
        )
        
        step += 1
    
    logger.info(f"Training completed. Total steps: {step}")
    actor_model.save('grpo-countdown-checkpoint', adapter_name=ADAPTER_NAME)
    swanlab.finish()


if __name__ == '__main__':
    main()
