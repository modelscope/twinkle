"""
GRPO LoRA Training Script for GPU (CUDA)

This script tests the twinkle RL training capabilities on GPU:
1. TransformersModel backend
2. VLLMSampler / TorchSampler integration
3. GRPOLoss and advantage computation
4. Weight synchronization between model and sampler

Based on lora_npu.py, adapted for CUDA GPU environment.

Usage:
    # Basic test with Transformers backend (local mode, no Ray)
    CUDA_VISIBLE_DEVICES=0 TWINKLE_MODE=local python lora_gpu.py

    # Test with multiple GPUs (Ray mode)
    CUDA_VISIBLE_DEVICES=0,1 TWINKLE_MODE=ray python lora_gpu.py

    # Use VLLMSampler (requires more GPU memory)
    TWINKLE_USE_TORCH_SAMPLER=0 python lora_gpu.py

    # Debug mode
    TWINKLE_DEBUG=1 python lora_gpu.py

Environment Variables:
    TWINKLE_MODEL_ID: Model path (default: Qwen/Qwen3-0.6B)
    TWINKLE_MAX_LENGTH: Max sequence length (default: 2048)
    TWINKLE_MAX_STEPS: Max training steps (default: 3)
    TWINKLE_USE_REF_MODEL: Use reference model for KL (default: 0)
    TWINKLE_USE_TORCH_SAMPLER: Use TorchSampler instead of VLLMSampler (default: 1)
    TWINKLE_DEBUG: Enable debug logging (default: 0)
    TWINKLE_MODE: 'local' or 'ray' (default: local)

Test Results (as of 2026-01-30):
    - TransformersModel + TorchSampler: PASS
    - VLLMSampler sampling: PASS  
    - VLLMSampler LoRA weight sync: IN PROGRESS (needs more debugging)
"""
import numpy as np
from peft import LoraConfig
import os
import sys

# Add twinkle src to path for development
twinkle_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if os.path.exists(twinkle_src):
    sys.path.insert(0, twinkle_src)

import twinkle
from twinkle import DeviceMesh, get_device_placement
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.infra import DeviceGroup, remote_function, remote_class
from twinkle.model import TransformersModel
from twinkle.reward import MathReward
from twinkle.sampler import VLLMSampler, TorchSampler
from twinkle.sampler.types import SamplingParams
from twinkle.weight_loader import NativeLoader
from twinkle.rl import compute_advantages

# Environment variable setup
os.environ.setdefault('TRUST_REMOTE_CODE', '1')
os.environ.setdefault('TWINKLE_SEED', '42')
os.environ.setdefault('TWINKLE_FULL_DETERMINISM', '1')

# Training configuration
use_ref_model = os.environ.get('TWINKLE_USE_REF_MODEL', '0') != '0'
use_torch_sampler = os.environ.get('TWINKLE_USE_TORCH_SAMPLER', '1') != '0'  # Default to TorchSampler for easier testing
num_generations = 8
kl_beta = 0.0
max_length = int(os.environ.get('TWINKLE_MAX_LENGTH', '2048'))
model_path = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3-0.6B')
debug_mode = os.environ.get('TWINKLE_DEBUG', '0') != '0'
run_mode = os.environ.get('TWINKLE_MODE', 'local')  # 'local' or 'ray'

# Global device meshes (will be set in train())
actor_device_mesh = None
ref_device_mesh = None


def build_template_kwargs(include_model_id: bool = False):
    kwargs = {}
    if include_model_id:
        kwargs['model_id'] = model_path
    if max_length > 0:
        kwargs['max_length'] = max_length
        kwargs['truncation_strategy'] = 'right'
    return kwargs


def parse_device_config():
    """Parse GPU device configuration from environment."""
    visible_devices_env = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices_env:
        visible_devices = [d for d in visible_devices_env.split(',') if d.strip()]
        nproc_per_node = len(visible_devices)
    else:
        # Try to detect available GPUs
        try:
            import torch
            nproc_per_node = torch.cuda.device_count()
        except:
            nproc_per_node = 1
    
    def _parse_ranks_env(name: str):
        raw = os.environ.get(name)
        if not raw:
            return None
        ranks = [int(v.strip()) for v in raw.split(',') if v.strip()]
        return ranks or None
    
    actor_ranks = _parse_ranks_env('TWINKLE_ACTOR_RANKS')
    ref_ranks = _parse_ranks_env('TWINKLE_REF_RANKS')
    
    if actor_ranks is None:
        actor_size = int(os.environ.get('TWINKLE_ACTOR_SIZE', str(nproc_per_node)))
        actor_ranks = list(range(actor_size))
    
    if ref_ranks is None and use_ref_model:
        ref_size = int(os.environ.get('TWINKLE_REF_SIZE', '1'))
        ref_start = (max(actor_ranks) + 1) if actor_ranks else 0
        ref_ranks = list(range(ref_start, ref_start + ref_size))
    
    return nproc_per_node, actor_ranks, ref_ranks


def create_device_groups(actor_ranks, ref_ranks):
    """Create device groups for actor and reference models."""
    device_groups = [
        DeviceGroup(
            name='actor',
            ranks=actor_ranks,
            device_type='cuda',  # GPU
        ),
    ]
    
    if use_ref_model and ref_ranks:
        device_groups.append(
            DeviceGroup(
                name='ref',
                ranks=ref_ranks,
                device_type='cuda',
            )
        )
    
    return device_groups


def create_device_meshes(actor_ranks, ref_ranks):
    """Create device meshes for actor and reference models."""
    actor_mesh = DeviceMesh(
        device_type='cuda',
        mesh=np.array([len(actor_ranks)]),
        mesh_dim_names=('dp',),
    )
    
    ref_mesh = DeviceMesh(
        device_type='cuda',
        mesh=np.array([len(ref_ranks) if ref_ranks is not None else 0]),
        mesh_dim_names=('dp',),
    )
    
    return actor_mesh, ref_mesh


def get_eos_token_ids():
    """Get EOS token IDs from tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        eos_ids = tokenizer.eos_token_id
        if eos_ids is None:
            return []
        elif isinstance(eos_ids, (list, tuple)):
            return list(eos_ids)
        else:
            return [eos_ids]
    except Exception as e:
        print(f'[WARN] Failed to get EOS token IDs: {e}')
        return []


def get_sampling_params(eos_token_ids) -> SamplingParams:
    """Create SamplingParams for generation."""
    return SamplingParams(
        max_tokens=128,
        temperature=1.0,
        top_p=0.95,
    )


def debug_print_rollout(step, trajectories, ground_truths, rewards=None):
    """Debug helper that prints rollout intermediates."""
    if not debug_mode:
        return

    # Extract prediction content (last message of the first sample)
    try:
        pred_msg = trajectories[0]['messages'][-1]['content'] if trajectories else None
    except (KeyError, IndexError, TypeError):
        pred_msg = None

    # Extract prompt and ground truth (first sample)
    try:
        prompt_msg = ground_truths[0]['messages'][0]['content'] if ground_truths else None
        gt_msg = ground_truths[0]['messages'][-1]['content'] if ground_truths else None
    except (KeyError, IndexError, TypeError):
        prompt_msg, gt_msg = None, None

    # Print prompt, prediction, and ground truth
    print(
        f'[DEBUG][step {step}] prompt={prompt_msg[:100] if prompt_msg else None}... | '
        f'pred={pred_msg[:100] if pred_msg else None}... | '
        f'gt={gt_msg[:100] if gt_msg else None}...',
        flush=True,
    )

    # Print reward statistics
    if rewards is not None and isinstance(rewards, (list, tuple)) and rewards:
        rewards_np = np.array(rewards, dtype=np.float32)
        print(
            f'[DEBUG][step {step}] rewards: n={len(rewards)}, '
            f'min={rewards_np.min():.4f}, mean={rewards_np.mean():.4f}, max={rewards_np.max():.4f}',
            flush=True,
        )


@remote_class()
class ActorGroup:
    """Actor group containing sampler and model for RL training."""
    
    def __init__(self, engine_args=None, lora_config=None, adapter_name=None, **kwargs):
        global actor_device_mesh
        
        if use_torch_sampler:
            self.sampler = TorchSampler(
                model_path,
                device_mesh=actor_device_mesh,
            )
        else:
            if engine_args is None:
                raise ValueError("engine_args is required for VLLMSampler.")
            self.sampler = VLLMSampler(
                model_path,
                engine_args=engine_args,
                device_mesh=actor_device_mesh,
            )
        self.sampler.add_adapter_to_sampler(adapter_name, lora_config)
        self.sampler.set_template('Template', adapter_name=adapter_name, **build_template_kwargs(include_model_id=True))
        
        self.model = TransformersModel(
            model_id=model_path, 
            remote_group='actor', 
            device_mesh=actor_device_mesh
        )
        self.model.add_adapter_to_model(adapter_name, lora_config)
        
        self.model.set_loss(
            'GRPOLoss',
            epsilon=0.2,
            beta=kl_beta,
            num_generations=num_generations,
        )
        
        self.model.set_optimizer('AdamW', lr=1e-6)
        self.model.set_lr_scheduler('LinearLR')
        self.model.set_template('Template', **build_template_kwargs(include_model_id=False))
        self.model.set_processor('GRPOLossProcessor')
        
        self.adapter_name = adapter_name
        self.lora_config = lora_config
    
    @remote_function(collect='flatten')
    def sample(self, batch, sampling_params: SamplingParams = None):
        return self.sampler.sample(batch, sampling_params=sampling_params, adapter_name=self.adapter_name)
    
    @remote_function()
    def forward(self, inputs, **kwargs):
        outputs = self.model.forward(inputs=inputs, **kwargs)
        return outputs['logits']
    
    @remote_function()
    def forward_only(self, inputs, **kwargs):
        outputs = self.model.forward_only(inputs=inputs, **kwargs)
        return outputs['logits']
    
    @remote_function()
    def forward_backward(self, inputs, trajectories, ref_logits=None, old_logits=None, **kwargs):
        if old_logits is None:
            old_logits = self.model.forward_only(inputs=inputs, **kwargs)['logits']
        return self.model.forward_backward(
            inputs=inputs,
            trajectories=trajectories,
            ref_logits=ref_logits,
            old_logits=old_logits,
            **kwargs,
        )
    
    @remote_function()
    def step(self):
        return self.model.step()
    
    @remote_function()
    def zero_grad(self):
        return self.model.zero_grad()
    
    @remote_function()
    def lr_step(self):
        return self.model.lr_step()

def create_dataset():
    """Create math dataset for RL training."""
    dataset = Dataset(DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Template', **build_template_kwargs(include_model_id=True))
    dataset.map('CompetitionMathGRPOProcessor')
    dataset.check(batched=True)
    return dataset


def create_simple_dataset():
    """Create a simple synthetic dataset for testing (no external dependencies)."""
    # Simple math-like prompts for testing
    # Multiple samples per batch to test advantage computation with num_generations > 1
    samples = [
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful math assistant. Respond with only the final answer in the form \\boxed{...}.'},
                {'role': 'user', 'content': 'What is 2 + 2?'},
                {'role': 'assistant', 'content': ''},
            ],
            'user_data': [('solution', '\\boxed{4}')],
        },
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful math assistant. Respond with only the final answer in the form \\boxed{...}.'},
                {'role': 'user', 'content': 'What is 3 * 5?'},
                {'role': 'assistant', 'content': ''},
            ],
            'user_data': [('solution', '\\boxed{15}')],
        },
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful math assistant. Respond with only the final answer in the form \\boxed{...}.'},
                {'role': 'user', 'content': 'What is 10 - 3?'},
                {'role': 'assistant', 'content': ''},
            ],
            'user_data': [('solution', '\\boxed{7}')],
        },
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful math assistant. Respond with only the final answer in the form \\boxed{...}.'},
                {'role': 'user', 'content': 'What is 6 / 2?'},
                {'role': 'assistant', 'content': ''},
            ],
            'user_data': [('solution', '\\boxed{3}')],
        },
    ]
    return samples


def train_local():
    """Local mode training - single process, no Ray."""
    global actor_device_mesh, ref_device_mesh
    
    import torch
    
    print(f'[INFO] Starting GRPO training on GPU (LOCAL mode)')
    print(f'[INFO] Model: {model_path}')
    print(f'[INFO] Use torch sampler: {use_torch_sampler}')
    print(f'[INFO] Max length: {max_length}')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')
    
    # Create simple device mesh for single GPU
    actor_device_mesh = DeviceMesh(
        device_type='cuda',
        mesh=np.array([1]),
        mesh_dim_names=('dp',),
    )
    ref_device_mesh = actor_device_mesh
    
    # Initialize twinkle in local mode
    twinkle.initialize(mode='local', nproc_per_node=1)
    
    # Use simple dataset for testing
    samples = create_simple_dataset()
    
    eos_token_ids = get_eos_token_ids()
    
    lora_config = LoraConfig(
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        r=8,
        lora_alpha=16,
    )
    
    # Create sampler
    print(f'[INFO] Creating sampler...', flush=True)
    if use_torch_sampler:
        from twinkle.sampler import TorchSampler
        sampler = TorchSampler(
            model_path,
            device_mesh=actor_device_mesh,
        )
    else:
        from twinkle.sampler import VLLMSampler
        engine_args = {
            'model': model_path,
            'enable_lora': True,
            'max_loras': 1,
            'max_lora_rank': 64,
            'max_model_len': max_length,
            'gpu_memory_utilization': 0.5,
            'trust_remote_code': True,
        }
        sampler = VLLMSampler(
            model_path,
            engine_args=engine_args,
            device_mesh=actor_device_mesh,
        )
    
    sampler.add_adapter_to_sampler('default', lora_config)
    sampler.set_template('Template', adapter_name='default', **build_template_kwargs(include_model_id=True))
    
    # Create model
    print(f'[INFO] Creating model...', flush=True)
    model = TransformersModel(
        model_id=model_path, 
        device_mesh=actor_device_mesh,
        trust_remote_code=True,
    )
    model.add_adapter_to_model('default', lora_config)
    
    model.set_loss(
        'GRPOLoss',
        epsilon=0.2,
        beta=kl_beta,
        num_generations=num_generations,
    )
    
    model.set_optimizer('AdamW', lr=1e-6)
    model.set_lr_scheduler('LinearLR')
    model.set_template('Template', **build_template_kwargs(include_model_id=True))
    model.set_processor('GRPOLossProcessor')
    
    # Create reward function
    reward = MathReward()
    
    # Create weight loader for syncing
    weight_loader = NativeLoader()
    
    step = 0
    max_steps = int(os.environ.get('TWINKLE_MAX_STEPS', '10'))
    
    print(f'[INFO] Starting training loop for {max_steps} steps')
    
    # Training loop
    for batch_idx in range(max_steps):
        step += 1
        print(f'\n[step {step}] ========== Starting iteration ==========', flush=True)
        
        # Use batch of 2 samples (num_generations=2 means 2 samples per prompt)
        # This allows proper advantage computation
        batch_start = (batch_idx * num_generations) % len(samples)
        batch_list = []
        for i in range(num_generations):
            sample_idx = (batch_start + i) % len(samples)
            batch_list.append(samples[sample_idx].copy())
        ground_truths = [b.copy() for b in batch_list]
        
        sampling_params = get_sampling_params(eos_token_ids)
        
        # Sample from policy
        print(f'[step {step}] Sampling...', flush=True)
        sample_response = sampler.sample(batch_list, sampling_params=sampling_params, adapter_name='default')
        
        # Convert sample response to trajectories
        trajectories = []
        for i, seq in enumerate(sample_response.sequences):
            # Decode the tokens to get the response text
            response_text = sampler.decode_response(seq.tokens, adapter_name='default')
            
            # Create trajectory with the sampled response
            src_batch = batch_list[i % len(batch_list)]
            traj = {
                'messages': [
                    src_batch['messages'][0],  # Keep system message if present
                    src_batch['messages'][1] if len(src_batch['messages']) > 1 else {'role': 'user', 'content': ''},
                    {'role': 'assistant', 'content': response_text},
                ],
                'user_data': src_batch.get('user_data', []),
            }
            trajectories.append(traj)
        
        print(f'[step {step}] Sampled {len(trajectories)} trajectories', flush=True)
        if debug_mode and trajectories:
            print(f'[step {step}] Sample response: {trajectories[0]["messages"][-1]["content"][:200]}...', flush=True)

        # Compute rewards
        print(f'[step {step}] Computing rewards...', flush=True)
        rewards = reward.calculate(trajectories, ground_truths)
        print(f'[step {step}] Rewards: {rewards}', flush=True)

        # Compute advantages
        # For single sample, use batch normalization
        if len(rewards) < num_generations:
            advantages = compute_advantages(rewards, num_generations=1, scale='batch')
        else:
            advantages = compute_advantages(rewards, num_generations=num_generations)
        
        for trajectory, advantage in zip(trajectories, advantages.tolist()):
            trajectory['advantages'] = advantage
        
        print(f'[step {step}] Advantages: {advantages.tolist()}', flush=True)

        # Debug print
        debug_print_rollout(step, trajectories, ground_truths, rewards=rewards)

        # Get old logits (for importance sampling ratio)
        print(f'[step {step}] Computing old logits...', flush=True)
        old_outputs = model.forward_only(inputs=trajectories)
        old_logits = old_outputs['logits'] if isinstance(old_outputs, dict) else old_outputs.logits

        # Forward-backward pass
        print(f'[step {step}] Forward-backward...', flush=True)
        loss = model.forward_backward(
            inputs=trajectories,
            trajectories=trajectories,
            old_logits=old_logits,
            ref_logits=None,
        )
        
        print(f'[step {step}] loss: {loss}', flush=True)
        
        # Optimizer step
        model.step()
        model.zero_grad()
        model.lr_step()
        
        # Sync weights to sampler
        print(f'[step {step}] Syncing weights...', flush=True)
        weight_loader(model, sampler, 'default')
        
        if max_steps and step >= max_steps:
            break
    
    print(f'\n[INFO] Training completed after {step} steps')


def train_ray():
    """Ray mode training - distributed with Ray."""
    global actor_device_mesh, ref_device_mesh
    
    nproc_per_node, actor_ranks, ref_ranks = parse_device_config()
    
    print(f'[INFO] Starting GRPO training on GPU (RAY mode)')
    print(f'[INFO] Model: {model_path}')
    print(f'[INFO] Actor ranks: {actor_ranks}')
    print(f'[INFO] Ref ranks: {ref_ranks}')
    print(f'[INFO] Use torch sampler: {use_torch_sampler}')
    print(f'[INFO] Use ref model: {use_ref_model}')
    
    device_groups = create_device_groups(actor_ranks, ref_ranks)
    actor_device_mesh, ref_device_mesh = create_device_meshes(actor_ranks, ref_ranks)
    
    # Initialize twinkle with ray mode
    twinkle.initialize(mode='ray', groups=device_groups, nproc_per_node=nproc_per_node)
    
    # Use simple dataset for testing
    samples = create_simple_dataset()
    
    eos_token_ids = get_eos_token_ids()
    
    engine_args = {
        'model': model_path,
        'enable_lora': True,
        'max_loras': 1,
        'max_lora_rank': 64,
        'max_model_len': max_length,
        'gpu_memory_utilization': float(os.environ.get('TWINKLE_VLLM_GPU_MEMORY_UTILIZATION', '0.9')),
        'trust_remote_code': True,
    }
    
    lora_config = LoraConfig(
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        r=8,
        lora_alpha=16,
    )
    
    actor_group = ActorGroup(
        None if use_torch_sampler else engine_args,
        remote_group='actor',
        lora_config=lora_config,
        adapter_name='default',
    )
    
    ref_model = None
    if use_ref_model:
        ref_model = TransformersModel(
            model_id=model_path, 
            remote_group='ref', 
            device_mesh=ref_device_mesh
        )
        ref_model.set_processor('InputProcessor')
        ref_model.set_template('Template', **build_template_kwargs())
    
    reward = MathReward()
    
    print('Device placement:', get_device_placement())
    
    step = 0
    max_steps = int(os.environ.get('TWINKLE_MAX_STEPS', '5'))
    
    # Training loop
    for batch_idx in range(max_steps):
        step += 1
        print(f'[step {step}] Starting iteration', flush=True)
        
        # Use samples cyclically
        batch = samples[batch_idx % len(samples)]
        if isinstance(batch, dict):
            batch_list = [batch]
        else:
            batch_list = list(batch)
        ground_truths = batch_list.copy()
        
        sampling_params = get_sampling_params(eos_token_ids)
        
        # Sample from policy
        print(f'[step {step}] Sampling...', flush=True)
        trajectories = actor_group.sample(batch_list, sampling_params)
        if callable(trajectories):
            trajectories = trajectories()
        print(f'[step {step}] Sampled {len(trajectories)} trajectories', flush=True)

        # Get reference logits if using ref model
        ref_logits = None
        if use_ref_model and ref_model is not None:
            ref_outputs = ref_model.forward_only(inputs=trajectories)
            if callable(ref_outputs) and getattr(ref_outputs, '_is_lazy_collect', False):
                ref_outputs = ref_outputs()
            if isinstance(ref_outputs, list):
                ref_logits = [o['logits'] if isinstance(o, dict) else o.logits for o in ref_outputs]
            else:
                ref_logits = ref_outputs['logits'] if isinstance(ref_outputs, dict) else ref_outputs.logits
        
        # Compute rewards
        print(f'[step {step}] Computing rewards...', flush=True)
        rewards = reward.calculate(trajectories, ground_truths)
        if callable(rewards):
            rewards = rewards()
        print(f'[step {step}] Rewards: {rewards}', flush=True)

        # Compute advantages
        advantages = compute_advantages(rewards, num_generations=num_generations)
        for trajectory, advantage in zip(trajectories, advantages.tolist()):
            trajectory['advantages'] = advantage

        # Debug print
        debug_print_rollout(step, trajectories, ground_truths, rewards=rewards)

        # Forward-backward pass
        print(f'[step {step}] Forward-backward...', flush=True)
        loss = actor_group.forward_backward(trajectories, trajectories, ref_logits)
        if callable(loss):
            loss = loss()
        
        print(f'[step {step}] loss: {loss}', flush=True)
        
        # Optimizer step
        actor_group.step()
        actor_group.zero_grad()
        actor_group.lr_step()
        
        if max_steps and step >= max_steps:
            break
    
    print(f'[INFO] Training completed after {step} steps')


def train():
    """Main training entry point."""
    if run_mode == 'local':
        train_local()
    else:
        train_ray()


if __name__ == '__main__':
    train()
