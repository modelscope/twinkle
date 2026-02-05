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
from twinkle import remote_class, remote_function
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

# SwanLab is optional - only used if SWANLAB_API_KEY is set
USE_SWANLAB = 'SWANLAB_API_KEY' in os.environ
if USE_SWANLAB:
    import swanlab

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen2.5-3B-Instruct')
NUM_GPUS = int(os.environ.get('NUM_GPUS', 4))
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 1024))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
GRPO_EPSILON = float(os.environ.get('GRPO_EPSILON', 0.2))
GRPO_BETA = float(os.environ.get('GRPO_BETA', 0.0))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 2000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 8))
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
    if USE_SWANLAB:
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
            log_dict['train/reward_std'] = torch.tensor(self.rewards).std().item() if len(self.rewards) > 1 else 0.0
        
        if self.format_rewards:
            log_dict['train/rewards/Format/mean'] = sum(self.format_rewards) / len(self.format_rewards)
            log_dict['train/rewards/Format/std'] = torch.tensor(self.format_rewards).std().item() if len(self.format_rewards) > 1 else 0.0
        
        if self.accuracy_rewards:
            log_dict['train/rewards/CountdownORM/mean'] = sum(self.accuracy_rewards) / len(self.accuracy_rewards)
            log_dict['train/rewards/CountdownORM/std'] = torch.tensor(self.accuracy_rewards).std().item() if len(self.accuracy_rewards) > 1 else 0.0
        
        if self.completion_lengths:
            log_dict['train/completions/mean_length'] = sum(self.completion_lengths) / len(self.completion_lengths)
            log_dict['train/completions/min_length'] = min(self.completion_lengths)
            log_dict['train/completions/max_length'] = max(self.completion_lengths)
        
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
    
    dataset = Dataset(DatasetMeta("ms://zouxuhong/Countdown-Tasks-3to4", data_slice=range(50000)))
    dataset.set_template("Template", model_id=MODEL_ID, max_length=8192)
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


def log(msg):
    """Print message with timestamp."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _collect_sample_responses(results):
    """Custom collect function to merge multiple SampleResponse objects from DP workers.
    
    Args:
        results: List of SampleResponse from each DP worker.
        
    Returns:
        Merged SampleResponse with all sequences combined.
    """
    if not results:
        return SampleResponse(sequences=[])
    
    if len(results) == 1:
        return results[0]
    
    all_sequences = []
    for resp in results:
        if resp is not None and hasattr(resp, 'sequences'):
            all_sequences.extend(resp.sequences)
    
    return SampleResponse(sequences=all_sequences)


# ========== Hybrid Actor ==========
@remote_class()
class HybridModelSamplerActor:
    """Hybrid actor that fuses training model and sampler in same process.
    
    This simulates the Hybrid mode where:
    - Training model (TransformersModel) holds the real weights
    - vLLM Sampler starts with dummy/random weights
    - Weight sync happens via IPCWeightLoader (CUDA IPC + ZMQ)
    """

    def __init__(
        self, 
        model_id: str, 
        device_mesh: DeviceMesh = None,
        lora_config = None,
        adapter_name: str = 'default',
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 8,
        epsilon: float = 0.2,
        beta: float = 0.0,
        remote_group: str = None,
        **kwargs
    ):
        import torch
        rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        log(f"[Rank {rank}] Initializing HybridModelSamplerActor...")
        
        self.adapter_name = adapter_name
        self.model_id = model_id
        self.lora_config = lora_config  # Store for weight sync
        
        # Initialize sampler with real model weights (not dummy)
        # For LoRA training, vLLM loads base model weights, then we sync LoRA weights
        self.sampler = VLLMSampler(
            model_id=model_id,
            engine_args={
                'gpu_memory_utilization': 0.4,
                'max_model_len': 2048,
                'enforce_eager': True,
                'enable_sleep_mode': True,
                # Enable LoRA in vLLM
                'enable_lora': True,
                'max_lora_rank': 64,
            },
        )
        self.sampler.set_template(Template, model_id=model_id)
        log(f"[Rank {rank}] VLLMSampler initialized with real base weights")
        
        # Initialize training model with real weights
        self.model = TransformersModel(model_id=model_id, device_mesh=device_mesh)
        log(f"[Rank {rank}] TransformersModel initialized with real weights")
        
        # Add LoRA adapter
        if lora_config is not None:
            self.model.add_adapter_to_model(adapter_name, lora_config, 
                                            gradient_accumulation_steps=gradient_accumulation_steps)
        
        # Set optimizer
        self.model.set_optimizer('AdamW', lr=learning_rate, adapter_name=adapter_name)
        
        # Set lr scheduler - use LinearLR for simplicity
        self.model.set_lr_scheduler('LinearLR', adapter_name=adapter_name)
        
        # Set loss
        self.model.set_loss('GRPOLoss', adapter_name=adapter_name, epsilon=epsilon, beta=beta)
        
        # Set processor
        self.model.set_processor(InputProcessor, adapter_name=adapter_name)
        
        # Set template
        self.model.set_template('Template', model_id=model_id, adapter_name=adapter_name)
        
        log(f"[Rank {rank}] Model configured with LoRA, optimizer, scheduler, loss")
        
        # Initialize weight loader for Hybrid mode (CUDA IPC)
        self.weight_loader = IPCWeightLoader(
            model=self.model,
            sampler=self.sampler,
            bucket_size_mb=512,
        )
        log(f"[Rank {rank}] IPCWeightLoader initialized")
    
    @remote_function(dispatch='slice_dp', collect=_collect_sample_responses, lazy_collect=False)
    def sample(self, batch, sampling_params: SamplingParams, num_samples: int = 1):
        """Sample from the model."""
        return self.sampler.sample(batch, sampling_params, num_samples=num_samples)
    
    @remote_function()
    def wake_up(self):
        """Wake up the sampler."""
        self.sampler.wake_up()
    
    @remote_function()
    def sleep(self):
        """Put the sampler to sleep."""
        self.sampler.sleep()
    
    @remote_function()
    def load_weights(self):
        """Sync LoRA weights from model to sampler.
        
        Since vLLM loads base model weights during initialization (not using load_format='dummy'),
        we only need to sync LoRA weights with base_sync_done=True.
        """
        from dataclasses import asdict
        peft_config = asdict(self.lora_config) if self.lora_config else None
        # base_sync_done=True: vLLM has base model, only sync LoRA weights
        self.weight_loader.load_weights(
            adapter_name=self.adapter_name, 
            peft_config=peft_config,
            base_sync_done=True,
        )
    
    @remote_function()
    def forward_backward(self, inputs, trajectories=None, old_logps=None, **kwargs):
        """Forward and backward pass."""
        return self.model.forward_backward(
            inputs=inputs,
            adapter_name=self.adapter_name,
            trajectories=trajectories,
            old_logps=old_logps,
            **kwargs,
        )
    
    @remote_function()
    def clip_grad_and_step(self):
        """Clip gradients and step optimizer."""
        return self.model.clip_grad_and_step(adapter_name=self.adapter_name)
    
    @remote_function()
    def get_train_configs(self):
        """Get training configs."""
        return self.model.get_train_configs(adapter_name=self.adapter_name)
    
    @remote_function()
    def save(self, path: str):
        """Save model checkpoint."""
        self.model.save(path, adapter_name=self.adapter_name)


# ========== Main ==========
def main():
    # SwanLab setup (optional)
    if USE_SWANLAB:
        swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=True)
        swanlab.init(project="ms-swift", config={
            'model_id': MODEL_ID,
            'num_gpus': NUM_GPUS,
            'num_generations': NUM_GENERATIONS,
            'learning_rate': LEARNING_RATE,
            'grpo_beta': GRPO_BETA,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        })
    else:
        logger.info("SWANLAB_API_KEY not set, running without experiment tracking")
    
    # Hybrid mode: model and sampler on same GPUs
    device_groups = [
        DeviceGroup(name='hybrid', ranks=list(range(NUM_GPUS)), device_type='GPU', gpus_per_worker=1),
    ]
    device_mesh = DeviceMesh.from_sizes(world_size=NUM_GPUS, dp_size=NUM_GPUS)
    
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)
    logger.info(get_device_placement())
    
    lora_config = LoraConfig(
        target_modules="all-linear",
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    
    # Create hybrid actor with all configurations
    hybrid_actor = HybridModelSamplerActor(
        model_id=MODEL_ID,
        device_mesh=device_mesh,
        lora_config=lora_config,
        adapter_name=ADAPTER_NAME,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        epsilon=GRPO_EPSILON,
        beta=GRPO_BETA,
        remote_group='hybrid',
    )
    
    # Log training config
    train_configs = wait_result(hybrid_actor.get_train_configs())
    logger.info(f"Training configs: {train_configs}")
    
    # Dataset
    dataset = create_countdown_dataset()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        device_mesh=device_mesh,
        remote_group='hybrid',
        num_workers=0,
    )
    
    # Tokenizer
    model_path = HubOperation.download_model(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Advantage calculator
    advantage_fn = GRPOAdvantage()
    
    # Metrics
    metrics = TrainingMetrics()
    
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )
    
    logger.info(f"Starting training for {MAX_STEPS} steps")
    logger.info(f"Config: batch_size={BATCH_SIZE}, num_generations={NUM_GENERATIONS}, "
                f"lr={LEARNING_RATE}, beta={GRPO_BETA}, epsilon={GRPO_EPSILON}")
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
            sync_start = time.perf_counter()
            wait_result(hybrid_actor.load_weights())
            metrics.weight_sync_time = time.perf_counter() - sync_start
        
        # ========== 2. Generate samples ==========
        wait_result(hybrid_actor.wake_up())
        
        gen_start = time.perf_counter()
        sample_response = wait_result(
            hybrid_actor.sample(prompts, sampling_params, num_samples=NUM_GENERATIONS)
        )
        metrics.generate_time = time.perf_counter() - gen_start
        
        wait_result(hybrid_actor.sleep())
        
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
        
        # Debug: print sample completions and rewards for first step
        if step == 0:
            logger.info(f"=== Debug: Step {step} sample completions ===")
            for i, traj in enumerate(trajectories[:3]):  # Print first 3
                messages = traj.get('messages', [])
                completion = ""
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        completion = msg.get('content', '')[:200]  # First 200 chars
                        break
                logger.info(f"Sample {i}: format_reward={format_rewards[i]}, acc_reward={accuracy_rewards[i]}")
                logger.info(f"  Completion: {completion}...")
            logger.info(f"Total rewards stats: mean={sum(total_rewards)/len(total_rewards):.4f}, "
                       f"format_mean={sum(format_rewards)/len(format_rewards):.4f}, "
                       f"acc_mean={sum(accuracy_rewards)/len(accuracy_rewards):.4f}")
        
        # ========== 5. Compute advantages and add to trajectories ==========
        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group')
        
        # Add advantages to trajectories (GRPOLoss extracts from trajectory['advantages'])
        for i, traj in enumerate(trajectories):
            traj['advantages'] = float(advantages[i])
        
        # Check if all advantages are zero (frac_reward_zero_std indicator)
        frac_zero_std = 1.0 if all(abs(adv) < 1e-8 for adv in advantages) else 0.0
        
        # Skip if all advantages are zero
        if frac_zero_std == 1.0:
            logger.info(f"Step {step}: All advantages are zero, skipping training")
            step += 1
            continue
        
        # ========== 6. Training step ==========
        loss = wait_result(hybrid_actor.forward_backward(
            inputs=trajectories,
            trajectories=trajectories,
            old_logps=old_logps_list,
        ))
        
        grad_norm = wait_result(hybrid_actor.clip_grad_and_step())
        
        metrics.loss = float(loss) if loss else 0.0
        metrics.grad_norm = float(grad_norm) if grad_norm else 0.0
        
        # ========== 7. Log metrics ==========
        log_dict = metrics.to_log_dict(step)
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        
        if USE_SWANLAB:
            swanlab.log(log_dict)
        
        logger.info(
            f"Step {step}: loss={metrics.loss:.6f}, grad_norm={metrics.grad_norm:.4f}, "
            f"reward={log_dict.get('train/reward', 0):.4f}, "
            f"format={log_dict.get('train/rewards/Format/mean', 0):.2f}, "
            f"accuracy={log_dict.get('train/rewards/CountdownORM/mean', 0):.2f}, "
            f"completion_len={log_dict.get('train/completions/mean_length', 0):.1f}"
        )
        
        step += 1
    
    logger.info(f"Training completed. Total steps: {step}")
    wait_result(hybrid_actor.save('grpo-countdown-checkpoint'))
    if USE_SWANLAB:
        swanlab.finish()


if __name__ == '__main__':
    main()
